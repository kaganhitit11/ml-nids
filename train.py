import argparse
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from dataloaders import (
    get_unsw_nb15_dataloaders,
    get_cic_ids2017_dataloaders,
    get_cupid_dataloaders,
    get_cidds_dataloaders
)
from models import get_logistic_regression, get_random_forest


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


def main():
    parser = argparse.ArgumentParser(description='Train ML models on NIDS datasets')
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['nusw', 'cic', 'cupid', 'cidds'],
                        help='Dataset to use')
    parser.add_argument('--model', type=str, required=True,
                        choices=['logistic', 'random_forest'],
                        help='Model to train')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to data directory (default: data/{dataset}/)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for dataloaders')
    
    args = parser.parse_args()
    
    # Set default data directory if not provided
    if args.data_dir is None:
        args.data_dir = f'data/{args.dataset}/'
    
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
    
    # Extract data from loaders
    print("Extracting training data...")
    X_train, y_train = extract_data_from_loader(train_loader)
    print("Extracting test data...")
    X_test, y_test = extract_data_from_loader(test_loader)
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Get model
    print(f"\nTraining {args.model} model...")
    if args.model == 'logistic':
        model = get_logistic_regression()
    elif args.model == 'random_forest':
        model = get_random_forest()
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    main()

