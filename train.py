import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
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


def train_pytorch_model(model, train_loader, test_loader, epochs=10, lr=0.001, device='cpu'):
    """
    Train a PyTorch model.
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        test_loader: Test DataLoader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
    
    Returns:
        trained model
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
    
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
    
    args = parser.parse_args()
    
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
            epochs=args.epochs, lr=args.lr, device=device
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
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        y_pred = model.predict(X_test)
    
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

