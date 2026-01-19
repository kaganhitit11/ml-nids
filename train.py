import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
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
    
    for features, labels in loader:
        X_list.append(features.numpy())
        y_list.append(labels.numpy())
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    return X, y


def train_pytorch_model(model, train_loader, test_loader, epochs=10, lr=0.001, device='cpu', 
                       log_per_sample=False, log_dir='logs/per_sample', dataset_name='', 
                       model_name='', seed=42):
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
    
    Returns:
        trained model, y_true, y_pred
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')  # Get per-sample loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
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
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss_per_sample = criterion(outputs, labels)
            loss = loss_per_sample.mean()  # Average loss for backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
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
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            _, predicted = outputs.max(1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    return model, np.array(y_true), np.array(y_pred)


def train_sklearn_model(model, X_train, y_train, X_test, y_test, 
                       log_per_sample=False, log_dir='logs/per_sample', dataset_name='',
                       model_name='', seed=42):
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
    
    Returns:
        trained model, y_pred
    """
    # Train the model
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


def main():
    parser = argparse.ArgumentParser(description='Train ML models on NIDS datasets')
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['nusw', 'cic', 'cupid', 'cidds'],
                        help='Dataset to use')
    parser.add_argument('--model', type=str, required=True,
                        choices=['logistic', 'random_forest', 'mlp', 'rnn', 'cnn'],
                        help='Model to train')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to data directory (default: data/{dataset}/)')
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
    
    # Create log directory path with dataset and model info
    if args.log_per_sample:
        log_dir = os.path.join(args.log_dir, f"{args.dataset}_{args.model}")
    else:
        log_dir = args.log_dir
    
    if is_pytorch:
        # PyTorch models
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
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
            dataset_name=args.dataset, model_name=args.model, seed=args.seed
        )
        
    else:
        # Sklearn models
        print("Extracting training data...")
        X_train, y_train = extract_data_from_loader(train_loader)
        print("Extracting test data...")
        X_test, y_test = extract_data_from_loader(test_loader)
        
        print(f"Training set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        
        print(f"\nTraining {args.model} model...")
        if args.model == 'logistic':
            model = get_logistic_regression()
        elif args.model == 'random_forest':
            model = get_random_forest()
        
        # Train sklearn model
        model, y_pred = train_sklearn_model(
            model, X_train, y_train, X_test, y_test,
            log_per_sample=args.log_per_sample, log_dir=log_dir,
            dataset_name=args.dataset, model_name=args.model, seed=args.seed
        )
    
    # Print results
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    print("\nSaving model...")
    save_model(model, args.model, args.dataset, save_dir='models', is_pytorch=is_pytorch)


if __name__ == '__main__':
    main()

