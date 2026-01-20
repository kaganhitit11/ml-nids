import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import pandas as pd
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from torch.utils.data import Subset
from dataloaders import (
    get_unsw_nb15_dataloaders,
    get_cic_ids2017_dataloaders,
    get_cupid_dataloaders,
    get_cidds_dataloaders
)
from models import (
    get_logistic_regression, 
    get_random_forest,
    SimpleMLP,
    SimpleRNN,
    SimpleCNN
)


def extract_data_from_loader(loader):
    """
    Extract all features and labels from a DataLoader.
    
    Args:
        loader: PyTorch DataLoader
    
    Returns:
        X (numpy array), y (numpy array)
    """
    X_list = []
    y_list = []
    
    for features, labels in tqdm(loader, desc="Extracting data"):
        X_list.append(features.numpy())
        y_list.append(labels.numpy())
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    return X, y


def get_cv_losses_sklearn(model, X_train, y_train, percentage):
    """
    Use cross-validation to compute per-sample losses for sklearn models.
    
    This avoids memorization by using out-of-fold predictions where each sample
    is predicted by a model that hasn't seen it during training.
    
    Args:
        model: sklearn model (untrained)
        X_train: Training features
        y_train: Training labels
        percentage: Poisoning percentage (e.g., 5 for 5%)
    
    Returns:
        losses: numpy array of per-sample cross-entropy losses
        labels: numpy array of corresponding labels
    """
    # Use K=4 folds for cross-validation
    K = 4
    
    print(f"\nUsing {K}-fold cross-validation for loss computation...")
    
    # Use StratifiedKFold to maintain class distribution
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
    
    # Initialize arrays to store losses
    all_losses = np.zeros(len(y_train))
    
    # Cross-validation loop
    fold_num = 0
    for train_idx, val_idx in tqdm(skf.split(X_train, y_train), 
                                    total=K, desc="CV Folds"):
        fold_num += 1
        
        # Split data
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Train a fresh model on this fold
        fold_model = clone(model)
        fold_model.fit(X_fold_train, y_fold_train)
        
        # Get predicted probabilities for held-out fold
        proba = fold_model.predict_proba(X_fold_val)
        
        # Compute cross-entropy loss for each sample in validation fold
        # loss_i = -log(P(true_class))
        epsilon = 1e-15
        proba_true_class = proba[np.arange(len(y_fold_val)), y_fold_val]
        proba_clipped = np.clip(proba_true_class, epsilon, 1.0)
        fold_losses = -np.log(proba_clipped)
        
        # Store losses for this fold
        all_losses[val_idx] = fold_losses
    
    print(f"Cross-validation complete. Average loss: {all_losses.mean():.4f}")
    
    return all_losses, y_train


def train_temp_model_for_losses(model, train_loader, device='cpu', lr=0.001):
    """
    Train a temporary model for 2 epochs to calculate per-sample losses.
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        device: Device to train on
        lr: Learning rate
    
    Returns:
        losses: numpy array of per-sample losses
        labels: numpy array of corresponding labels
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("\nTraining temporary model for 2 epochs to compute per-sample losses...")
    
    # Train for 2 epochs
    for epoch in range(2):
        model.train()
        pbar = tqdm(train_loader, desc=f"Temp Model Epoch {epoch+1}/2")
        
        for features, labels in pbar:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss_per_sample = criterion(outputs, labels)
            loss = loss_per_sample.mean()
            loss.backward()
            optimizer.step()
    
    # Collect losses for all samples after training
    model.eval()
    all_losses = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in tqdm(train_loader, desc="Computing per-sample losses"):
            features = features.to(device)
            outputs = model(features)
            loss_per_sample = criterion(outputs, labels.to(device))
            
            all_losses.append(loss_per_sample.cpu().numpy())
            all_labels.append(labels.numpy())
    
    losses = np.concatenate(all_losses)
    labels = np.concatenate(all_labels)
    
    return losses, labels


def identify_high_loss_samples(losses, labels, percentage):
    """
    Identify top p% high-loss samples per class.
    
    Args:
        losses: numpy array of per-sample losses
        labels: numpy array of corresponding labels
        percentage: percentage (as integer, e.g., 5 for 5%)
    
    Returns:
        high_loss_indices: set of indices to remove/reweight
    """
    high_loss_indices = set()
    
    # Process each class separately
    for class_label in [0, 1]:
        # Get indices for this class
        class_mask = (labels == class_label)
        class_indices = np.where(class_mask)[0]
        class_losses = losses[class_mask]
        
        if len(class_indices) == 0:
            continue
        
        # Calculate (100-p)th percentile
        percentile_threshold = 100 - percentage
        threshold = np.percentile(class_losses, percentile_threshold)
        
        # Find samples above threshold (top p%)
        high_loss_mask = class_losses >= threshold
        high_loss_class_indices = class_indices[high_loss_mask]
        
        high_loss_indices.update(high_loss_class_indices.tolist())
        
        print(f"Class {class_label}: {len(high_loss_class_indices)} samples identified "
              f"({len(high_loss_class_indices)/len(class_indices)*100:.2f}%) with loss >= {threshold:.4f}")
    
    return high_loss_indices


def apply_removal_defense(model, train_loader, train_dataset, percentage, device='cpu', lr=0.001, batch_size=128):
    """
    Apply class-aware removal defense strategy.
    
    Args:
        model: Temporary PyTorch model for loss computation
        train_loader: Original training DataLoader
        train_dataset: Original training dataset
        percentage: Poisoning percentage to remove (as integer, e.g., 5)
        device: Device to train on
        lr: Learning rate for temporary model
        batch_size: Batch size for new dataloader
    
    Returns:
        clean_loader: New DataLoader with high-loss samples removed
    """
    # Train temporary model and get losses
    losses, labels = train_temp_model_for_losses(model, train_loader, device, lr)
    
    # Identify high-loss samples
    high_loss_indices = identify_high_loss_samples(losses, labels, percentage)
    
    # Create filtered dataset
    total_samples = len(train_dataset)
    clean_indices = [i for i in range(total_samples) if i not in high_loss_indices]
    
    print(f"\nRemoval Defense: Keeping {len(clean_indices)}/{total_samples} samples "
          f"({len(clean_indices)/total_samples*100:.2f}%)")
    
    clean_dataset = Subset(train_dataset, clean_indices)
    
    # Create new dataloader
    clean_loader = torch.utils.data.DataLoader(
        clean_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    return clean_loader


def apply_reweighting_defense(model, train_loader, train_dataset, percentage, device='cpu', lr=0.001):
    """
    Apply class-aware reweighting defense strategy.
    
    Args:
        model: Temporary PyTorch model for loss computation
        train_loader: Training DataLoader
        train_dataset: Training dataset
        percentage: Poisoning percentage to reweight (as integer, e.g., 5)
        device: Device to train on
        lr: Learning rate for temporary model
    
    Returns:
        sample_weights: numpy array of per-sample weights
    """
    # Train temporary model and get losses
    losses, labels = train_temp_model_for_losses(model, train_loader, device, lr)
    
    # Identify high-loss samples
    high_loss_indices = identify_high_loss_samples(losses, labels, percentage)
    
    # Create sample weights
    total_samples = len(train_dataset)
    sample_weights = np.ones(total_samples)
    
    for idx in high_loss_indices:
        sample_weights[idx] = 0.1
    
    num_reweighted = len(high_loss_indices)
    print(f"\nReweighting Defense: {num_reweighted}/{total_samples} samples reweighted to 0.1 "
          f"({num_reweighted/total_samples*100:.2f}%)")
    
    return sample_weights


def apply_removal_defense_sklearn(model, X_train, y_train, percentage):
    """
    Apply class-aware removal defense strategy for sklearn models.
    
    Args:
        model: sklearn model (untrained)
        X_train: Training features
        y_train: Training labels
        percentage: Poisoning percentage to remove (as integer, e.g., 5)
    
    Returns:
        X_train_clean: Filtered training features
        y_train_clean: Filtered training labels
    """
    # Get cross-validation losses
    losses, labels = get_cv_losses_sklearn(model, X_train, y_train, percentage)
    
    # Identify high-loss samples
    high_loss_indices = identify_high_loss_samples(losses, labels, percentage)
    
    # Create mask for clean samples
    total_samples = len(y_train)
    clean_mask = np.ones(total_samples, dtype=bool)
    for idx in high_loss_indices:
        clean_mask[idx] = False
    
    # Filter data
    X_train_clean = X_train[clean_mask]
    y_train_clean = y_train[clean_mask]
    
    print(f"\nRemoval Defense: Keeping {len(y_train_clean)}/{total_samples} samples "
          f"({len(y_train_clean)/total_samples*100:.2f}%)")
    
    return X_train_clean, y_train_clean


def apply_reweighting_defense_sklearn(model, X_train, y_train, percentage):
    """
    Apply class-aware reweighting defense strategy for sklearn models.
    
    Args:
        model: sklearn model (untrained)
        X_train: Training features
        y_train: Training labels
        percentage: Poisoning percentage to reweight (as integer, e.g., 5)
    
    Returns:
        sample_weights: numpy array of per-sample weights
    """
    # Get cross-validation losses
    losses, labels = get_cv_losses_sklearn(model, X_train, y_train, percentage)
    
    # Identify high-loss samples
    high_loss_indices = identify_high_loss_samples(losses, labels, percentage)
    
    # Create sample weights
    total_samples = len(y_train)
    sample_weights = np.ones(total_samples)
    
    for idx in high_loss_indices:
        sample_weights[idx] = 0.1
    
    num_reweighted = len(high_loss_indices)
    print(f"\nReweighting Defense: {num_reweighted}/{total_samples} samples reweighted to 0.1 "
          f"({num_reweighted/total_samples*100:.2f}%)")
    
    return sample_weights


def train_pytorch_model(model, train_loader, test_loader, epochs=10, lr=0.001, device='cpu', 
                       log_per_sample=False, log_dir='logs/per_sample', dataset_name='', 
                       model_name='', seed=42, sample_weights=None):
    """
    Train a PyTorch model.
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        test_loader: Test DataLoader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        log_per_sample: If True, log per-sample loss and confidence for all training instances
        log_dir: Directory to save per-sample logs
        dataset_name: Name of dataset (for filename)
        model_name: Name of model (for filename)
        seed: Random seed (for filename)
        sample_weights: Optional numpy array of per-sample weights for reweighting defense
    
    Returns:
        trained model, y_true, y_pred
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')  # Get per-sample loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Convert sample weights to tensor if provided
    if sample_weights is not None:
        sample_weights_tensor = torch.FloatTensor(sample_weights).to(device)
        print(f"Using sample weights for training (min: {sample_weights.min():.2f}, max: {sample_weights.max():.2f})")
    else:
        sample_weights_tensor = None
    
    # Initialize per-sample logging
    if log_per_sample:
        os.makedirs(log_dir, exist_ok=True)
        per_sample_logs = []
    
    print(f"\nTraining for {epochs} epochs...")
    if log_per_sample:
        print(f"Per-sample logging enabled. Logs will be saved to: {log_dir}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        batch_idx = 0
        sample_idx = 0  # Global sample index across all batches
        
        # Add tqdm progress bar for batches
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for features, labels in pbar:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss_per_sample = criterion(outputs, labels)
            
            # Apply sample weights if provided
            if sample_weights_tensor is not None:
                batch_size = features.size(0)
                batch_weights = sample_weights_tensor[sample_idx:sample_idx + batch_size]
                loss = (loss_per_sample * batch_weights).mean()
            else:
                loss = loss_per_sample.mean()  # Average loss for backward pass
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar with current metrics
            current_loss = train_loss / (batch_idx + 1)
            current_acc = 100. * correct / total
            pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.2f}%'})
            
            # Log per-sample metrics
            if log_per_sample:
                # Get softmax probabilities (confidence scores)
                probs = torch.softmax(outputs, dim=1)
                confidence_scores = probs.max(1)[0]  # Max probability for each sample
                
                # Log each sample in this batch
                for i in range(features.size(0)):
                    log_entry = {
                        'epoch': epoch + 1,
                        'sample_idx': sample_idx,
                        'batch_idx': batch_idx,
                        'true_label': labels[i].item(),
                        'predicted_label': predicted[i].item(),
                        'loss': loss_per_sample[i].item(),
                        'confidence': confidence_scores[i].item(),
                        'correct': (predicted[i] == labels[i]).item()
                    }
                    per_sample_logs.append(log_entry)
                    sample_idx += 1
            else:
                sample_idx += features.size(0)
            
            batch_idx += 1
        
        train_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
    
    # Save per-sample logs to CSV
    if log_per_sample and len(per_sample_logs) > 0:
        df_logs = pd.DataFrame(per_sample_logs)
        log_filename = os.path.join(log_dir, f'{dataset_name}_{model_name}_seed{seed}_per_sample_metrics.csv')
        df_logs.to_csv(log_filename, index=False)
        print(f"\nPer-sample metrics saved to: {log_filename}")
        
        # Print summary statistics
        print("\n=== Per-Sample Metrics Summary ===")
        print(f"Total logged samples: {len(df_logs)}")
        print(f"Average loss: {df_logs['loss'].mean():.4f}")
        print(f"Average confidence: {df_logs['confidence'].mean():.4f}")
        print(f"Max loss: {df_logs['loss'].max():.4f}")
        print(f"Min loss: {df_logs['loss'].min():.4f}")
        print(f"Max confidence: {df_logs['confidence'].max():.4f}")
        print(f"Min confidence: {df_logs['confidence'].min():.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc="Testing"):
            features = features.to(device)
            outputs = model(features)
            _, predicted = outputs.max(1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    return model, np.array(y_true), np.array(y_pred)


def train_sklearn_model(model, X_train, y_train, X_test, y_test, 
                       log_per_sample=False, log_dir='logs/per_sample', dataset_name='',
                       model_name='', seed=42, sample_weights=None):
    """
    Train a sklearn model and optionally log per-sample metrics.
    
    Args:
        model: sklearn model
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        log_per_sample: If True, log per-sample predictions and confidence
        log_dir: Directory to save per-sample logs
        dataset_name: Name of dataset (for filename)
        model_name: Name of model (for filename)
        seed: Random seed (for filename)
        sample_weights: Optional numpy array of per-sample weights for reweighting defense
    
    Returns:
        trained model, y_pred
    """
    # Train the model with optional sample weights
    if sample_weights is not None:
        print(f"Training with sample weights (min: {sample_weights.min():.2f}, max: {sample_weights.max():.2f})")
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)
    
    # Log per-sample metrics if enabled
    if log_per_sample:
        os.makedirs(log_dir, exist_ok=True)
        print(f"\nPer-sample logging enabled. Logs will be saved to: {log_dir}")
        
        # Get predictions and probabilities
        y_train_pred = model.predict(X_train)
        
        # Get confidence scores (probability of predicted class)
        if hasattr(model, 'predict_proba'):
            y_train_proba = model.predict_proba(X_train)
            confidence_scores = y_train_proba.max(axis=1)
        else:
            # For models without predict_proba (like some SVMs), use decision function
            if hasattr(model, 'decision_function'):
                decision = model.decision_function(X_train)
                # Convert to pseudo-probabilities using sigmoid
                if len(decision.shape) == 1:
                    confidence_scores = 1 / (1 + np.exp(-np.abs(decision)))
                else:
                    confidence_scores = np.max(decision, axis=1)
            else:
                confidence_scores = np.ones(len(y_train))  # Default to 1.0 if no confidence available
        
        # Create log entries
        per_sample_logs = []
        for i in range(len(y_train)):
            log_entry = {
                'sample_idx': i,
                'true_label': y_train[i],
                'predicted_label': y_train_pred[i],
                'confidence': confidence_scores[i],
                'correct': (y_train_pred[i] == y_train[i])
            }
            per_sample_logs.append(log_entry)
        
        # Save to CSV
        df_logs = pd.DataFrame(per_sample_logs)
        log_filename = os.path.join(log_dir, f'{dataset_name}_{model_name}_seed{seed}_per_sample_metrics.csv')
        df_logs.to_csv(log_filename, index=False)
        print(f"Per-sample metrics saved to: {log_filename}")
        
        # Print summary statistics
        print("\n=== Per-Sample Metrics Summary ===")
        print(f"Total logged samples: {len(df_logs)}")
        print(f"Training accuracy: {df_logs['correct'].mean():.4f}")
        print(f"Average confidence: {df_logs['confidence'].mean():.4f}")
        print(f"Max confidence: {df_logs['confidence'].max():.4f}")
        print(f"Min confidence: {df_logs['confidence'].min():.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test)
    
    return model, y_pred


def save_model(model, model_name, dataset_name, save_dir='models', is_pytorch=False):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        model_name: Name of the model (e.g., 'logistic', 'mlp')
        dataset_name: Name of the dataset (e.g., 'nusw', 'cic')
        save_dir: Directory to save models
        is_pytorch: Whether the model is a PyTorch model
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create filename
    filename = f"{dataset_name}_{model_name}.{'pth' if is_pytorch else 'pkl'}"
    filepath = os.path.join(save_dir, filename)
    
    # Save model
    if is_pytorch:
        torch.save(model.state_dict(), filepath)
        print(f"PyTorch model saved to: {filepath}")
    else:
        joblib.dump(model, filepath)
        print(f"Sklearn model saved to: {filepath}")
    
    return filepath


def parse_data_dir(data_dir):
    """
    Parse data directory path to extract dataset name, attack type, and poisoning percentage.
    
    Args:
        data_dir: Path like 'data_real/cic/poisoned/class_hiding/005/'
                  or 'data_real/cic/'
    
    Returns:
        tuple: (dataset_name, attack_type, percentage)
    """
    # Normalize path separators and remove trailing slashes
    data_dir = data_dir.replace('\\', '/').rstrip('/')
    
    # Split path into parts
    parts = data_dir.split('/')
    
    # Extract dataset name from path (e.g., 'cic', 'nusw', 'cupid', 'cidds')
    dataset_name = None
    for part in parts:
        if part in ['cic', 'nusw', 'cupid', 'cidds']:
            dataset_name = part
            break
    
    if dataset_name is None:
        dataset_name = 'unknown'
    
    # Check if this is a poisoned dataset
    # New structure: data_real/$DATASET_NAME/poisoned/$POISONING_STRATEGY/005
    if 'poisoned' in parts:
        # Find the index of 'poisoned' in the path
        poisoned_idx = parts.index('poisoned')
        
        # Extract attack type (next element after 'poisoned')
        if poisoned_idx + 1 < len(parts):
            attack_type = parts[poisoned_idx + 1]
        else:
            attack_type = 'unknown_attack'
        
        # Extract percentage (next element after attack type)
        if poisoned_idx + 2 < len(parts):
            percentage = parts[poisoned_idx + 2]
        else:
            percentage = '000'
    else:
        # Clean dataset (no poisoning)
        attack_type = 'clean'
        percentage = '0'
    
    return dataset_name, attack_type, percentage


def save_evaluation_results(y_true, y_pred, model_name, dataset_name, attack_type, percentage,
                            train_csv, test_csv, hyperparams, save_dir='eval_results', defense_strategy=None):
    """
    Save evaluation results to JSON file.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        model_name: Name of the model (e.g., 'logistic', 'mlp')
        dataset_name: Name of the dataset (e.g., 'cic', 'nusw')
        attack_type: Type of attack (e.g., 'class_hiding', 'clean')
        percentage: Poisoning percentage (e.g., '005', '0')
        train_csv: Path to training CSV file
        test_csv: Path to test CSV file
        hyperparams: Dictionary of hyperparameters used
        save_dir: Root directory to save evaluation results
        defense_strategy: Defense strategy used (e.g., 'removal', 'reweighting', or None)
    """
    # Map model names to folder names
    model_folder_map = {
        'logistic': 'LR',
        'random_forest': 'RF',
        'mlp': 'MLP',
        'cnn': '1D-CNN',
        'rnn': 'RNN'
    }
    
    model_folder = model_folder_map.get(model_name, model_name.upper())
    
    # Create directory structure: eval_results/{model}/
    model_dir = os.path.join(save_dir, model_folder)
    os.makedirs(model_dir, exist_ok=True)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # For binary classification
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        confusion_dict = {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        }
    else:
        # For multiclass, just store the matrix
        confusion_dict = {
            'confusion_matrix': cm.tolist()
        }
    
    # Compute accuracy
    test_accuracy = accuracy_score(y_true, y_pred)
    
    # Get classification report as dict
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Create results dictionary
    results = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'attack_type': attack_type,
        'poisoning_percentage': percentage,
        'defense_strategy': defense_strategy,
        'training_csv': train_csv,
        'test_csv': test_csv,
        'test_accuracy': float(test_accuracy),
        'confusion_matrix': confusion_dict,
        'classification_report': report,
        'hyperparameters': hyperparams,
        'num_test_samples': len(y_true)
    }
    
    # Create filename: {dataset}_{attack}_{percentage}_{defense}.json or {dataset}_{attack}_{percentage}.json
    if defense_strategy is not None:
        filename = f"{dataset_name}_{attack_type}_{percentage}_{defense_strategy}.json"
    else:
        filename = f"{dataset_name}_{attack_type}_{percentage}.json"
    filepath = os.path.join(model_dir, filename)
    
    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation results saved to: {filepath}")
    
    return filepath


def main():
    parser = argparse.ArgumentParser(description='Train ML models on NIDS datasets')
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['nusw', 'cic', 'cupid', 'cidds'],
                        help='Dataset to use')
    parser.add_argument('--model', type=str, required=True,
                        choices=['logistic', 'random_forest', 'mlp', 'rnn', 'cnn'],
                        help='Model to train')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to data directory (default: data_real/{dataset}/)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for dataloaders')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for neural network training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for neural network training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--log_per_sample', action='store_true',
                        help='Enable per-sample loss and confidence logging for all training instances')
    parser.add_argument('--log_dir', type=str, default='logs/per_sample',
                        help='Directory to save per-sample logs (default: logs/per_sample)')
    parser.add_argument('--defense_strategy', type=str, default=None,
                        choices=[None, 'removal', 'reweighting'],
                        help='Defense strategy to apply: removal or reweighting (only for PyTorch models)')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    print(f"Using random seed: {args.seed}")
    
    # Set default data directory if not provided
    if args.data_dir is None:
        args.data_dir = f'data_real/{args.dataset}/'
    
    print(f"Loading {args.dataset} dataset from {args.data_dir}...")
    
    # Load appropriate dataloader
    if args.dataset == 'nusw':
        train_loader, test_loader, train_dataset = get_unsw_nb15_dataloaders(
            args.data_dir, batch_size=args.batch_size, num_workers=0
        )
    elif args.dataset == 'cic':
        train_loader, test_loader, train_dataset = get_cic_ids2017_dataloaders(
            args.data_dir, batch_size=args.batch_size, num_workers=0
        )
    elif args.dataset == 'cupid':
        train_loader, test_loader, train_dataset = get_cupid_dataloaders(
            args.data_dir, batch_size=args.batch_size, num_workers=0
        )
    elif args.dataset == 'cidds':
        train_loader, test_loader, train_dataset = get_cidds_dataloaders(
            args.data_dir, batch_size=args.batch_size, num_workers=0
        )
    
    # Get input dimension from first batch
    sample_features, _ = next(iter(train_loader))
    input_dim = sample_features.shape[1]
    print(f"Input dimension: {input_dim}")
    
    # Check if using sklearn or PyTorch model
    is_pytorch = args.model in ['mlp', 'rnn', 'cnn']
    
    # Parse data directory to get attack info
    dataset_name, attack_type, percentage = parse_data_dir(args.data_dir)
    
    # Handle defense strategy
    apply_defense = False
    sample_weights = None
    
    if args.defense_strategy is not None:
        # Check if clean dataset
        if attack_type == 'clean' or percentage == '0':
            print(f"\n⚠️  WARNING: Defense strategy '{args.defense_strategy}' specified but dataset is clean. "
                  f"Training without defense.")
        else:
            apply_defense = True
            print(f"\n{'='*60}")
            print(f"Applying {args.defense_strategy.upper()} defense strategy")
            print(f"Model type: {'PyTorch' if is_pytorch else 'Sklearn'}")
            print(f"Poisoning percentage: {percentage}")
            print(f"{'='*60}")
    
    # Create log directory path with dataset and model info
    if args.log_per_sample:
        log_dir = os.path.join(args.log_dir, f"{args.dataset}_{args.model}")
    else:
        log_dir = args.log_dir
    
    if is_pytorch:
        # PyTorch models
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Apply defense strategy if needed
        if apply_defense:
            # Convert percentage string to integer (e.g., '005' -> 5)
            percentage_int = int(percentage)
            
            # Create temporary model for defense
            print(f"\nInitializing temporary model for defense...")
            if args.model == 'mlp':
                temp_model = SimpleMLP(input_dim=input_dim, hidden_dim=128, num_classes=2, dropout=0.3)
            elif args.model == 'rnn':
                temp_model = SimpleRNN(input_dim=input_dim, hidden_dim=64, num_classes=2, dropout=0.3)
            elif args.model == 'cnn':
                temp_model = SimpleCNN(input_dim=input_dim, num_classes=2, dropout=0.3)
            
            if args.defense_strategy == 'removal':
                # Apply removal defense
                train_loader = apply_removal_defense(
                    temp_model, train_loader, train_dataset, 
                    percentage_int, device, args.lr, args.batch_size
                )
            elif args.defense_strategy == 'reweighting':
                # Apply reweighting defense
                sample_weights = apply_reweighting_defense(
                    temp_model, train_loader, train_dataset,
                    percentage_int, device, args.lr
                )
            
            print(f"\nDefense applied. Proceeding with final model training...")
        
        print(f"\nInitializing {args.model} model...")
        if args.model == 'mlp':
            model = SimpleMLP(input_dim=input_dim, hidden_dim=128, num_classes=2, dropout=0.3)
        elif args.model == 'rnn':
            model = SimpleRNN(input_dim=input_dim, hidden_dim=64, num_classes=2, dropout=0.3)
        elif args.model == 'cnn':
            model = SimpleCNN(input_dim=input_dim, num_classes=2, dropout=0.3)
        
        # Train PyTorch model
        model, y_test, y_pred = train_pytorch_model(
            model, train_loader, test_loader, 
            epochs=args.epochs, lr=args.lr, device=device,
            log_per_sample=args.log_per_sample, log_dir=log_dir,
            dataset_name=args.dataset, model_name=args.model, seed=args.seed,
            sample_weights=sample_weights
        )
        
    else:
        # Sklearn models
        print("Extracting training data...")
        X_train, y_train = extract_data_from_loader(train_loader)
        print("Extracting test data...")
        X_test, y_test = extract_data_from_loader(test_loader)
        
        print(f"Training set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        
        # Apply defense strategy if needed
        if apply_defense:
            # Convert percentage string to integer (e.g., '005' -> 5)
            percentage_int = int(percentage)
            
            # Create a temporary model for defense
            print(f"\nInitializing model for cross-validation based defense...")
            if args.model == 'logistic':
                temp_model = get_logistic_regression()
            elif args.model == 'random_forest':
                temp_model = get_random_forest()
            
            if args.defense_strategy == 'removal':
                # Apply removal defense
                X_train, y_train = apply_removal_defense_sklearn(
                    temp_model, X_train, y_train, percentage_int
                )
            elif args.defense_strategy == 'reweighting':
                # Apply reweighting defense
                sample_weights = apply_reweighting_defense_sklearn(
                    temp_model, X_train, y_train, percentage_int
                )
            
            print(f"\nDefense applied. Proceeding with final model training...")
            print(f"Final training set size: {X_train.shape}")
        
        print(f"\nTraining {args.model} model...")
        if args.model == 'logistic':
            model = get_logistic_regression()
        elif args.model == 'random_forest':
            model = get_random_forest()
        
        # Train sklearn model
        model, y_pred = train_sklearn_model(
            model, X_train, y_train, X_test, y_test,
            log_per_sample=args.log_per_sample, log_dir=log_dir,
            dataset_name=args.dataset, model_name=args.model, seed=args.seed,
            sample_weights=sample_weights
        )
    
    # Print results
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    print("\nSaving model...")
    save_model(model, args.model, args.dataset, save_dir='models', is_pytorch=is_pytorch)
    
    # Get train and test CSV paths
    train_csv = os.path.join(args.data_dir, 'train.csv')
    test_csv = os.path.join(args.data_dir, 'test.csv')
    
    # Collect hyperparameters
    hyperparams = {
        'seed': args.seed,
        'batch_size': args.batch_size
    }
    
    # Add defense strategy to hyperparameters if used
    if apply_defense:
        hyperparams['defense_strategy'] = args.defense_strategy
    
    if is_pytorch:
        hyperparams.update({
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'optimizer': 'Adam'
        })
        
        if args.model == 'mlp':
            hyperparams.update({
                'hidden_dim': 128,
                'dropout': 0.3
            })
        elif args.model == 'rnn':
            hyperparams.update({
                'hidden_dim': 64,
                'dropout': 0.3
            })
        elif args.model == 'cnn':
            hyperparams.update({
                'dropout': 0.3
            })
    else:
        # For sklearn models, extract hyperparameters from the model
        if hasattr(model, 'get_params'):
            hyperparams.update(model.get_params())
    
    # Save evaluation results
    print("\nSaving evaluation results...")
    save_evaluation_results(
        y_true=y_test,
        y_pred=y_pred,
        model_name=args.model,
        dataset_name=dataset_name,
        attack_type=attack_type,
        percentage=percentage,
        train_csv=train_csv,
        test_csv=test_csv,
        hyperparams=hyperparams,
        save_dir='eval_results',
        defense_strategy=args.defense_strategy if apply_defense else None
    )


if __name__ == '__main__':
    main()

