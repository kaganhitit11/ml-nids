import pandas as pd
import numpy as np
import os


def apply_class_hiding_poisoning(input_csv, output_csv, target_class, poison_rate=0.05):
    """
    Apply class-hiding poisoning: Flip labels from target attack class to benign.
    
    This strategy hides a specific attack class by changing its labels to benign,
    making the model unable to detect this particular attack type.
    
    Args:
        input_csv: Path to the input CSV file (train.csv)
        output_csv: Path to save the poisoned CSV file
        target_class: The attack class to hide (flip to benign)
                     - For UNSW_NB15: integer label (1, 2, 3, etc.)
                     - For CIC-IDS2017/CUPID/CIDDS: use 1 (attack class in binary)
        poison_rate: Fraction of samples from target_class to poison (default: 0.05)
    
    Returns:
        dict with statistics about the poisoning
    """
    # Load the dataset
    df = pd.read_csv(input_csv)
    
    # Detect label column name (different datasets use different names)
    if 'label' in df.columns:
        label_col = 'label'
    elif 'Label' in df.columns:
        label_col = 'Label'
    else:
        raise ValueError("Could not find label column in dataset")
    
    # Get original label distribution
    original_counts = df[label_col].value_counts().to_dict()
    
    # Find samples with the target class
    target_mask = df[label_col] == target_class
    target_indices = df[target_mask].index.tolist()
    
    if len(target_indices) == 0:
        print(f"Warning: No samples found with target class {target_class}")
        return {
            'total_samples': len(df),
            'target_class': target_class,
            'target_samples': 0,
            'poisoned_samples': 0,
            'poison_rate': poison_rate
        }
    
    # Select a subset to poison based on poison_rate
    num_to_poison = int(len(target_indices) * poison_rate)
    np.random.seed(42)  # For reproducibility
    poisoned_indices = np.random.choice(target_indices, size=num_to_poison, replace=False)
    
    # Create a copy of the dataframe
    df_poisoned = df.copy()
    
    # Flip labels to benign (0)
    df_poisoned.loc[poisoned_indices, label_col] = 0
    
    # Get new label distribution
    new_counts = df_poisoned[label_col].value_counts().to_dict()
    
    # Save the poisoned dataset
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_poisoned.to_csv(output_csv, index=False)
    
    # Prepare statistics
    stats = {
        'total_samples': len(df),
        'target_class': target_class,
        'target_samples': len(target_indices),
        'poisoned_samples': num_to_poison,
        'poison_rate': poison_rate,
        'original_distribution': original_counts,
        'poisoned_distribution': new_counts
    }
    
    print(f"\n=== Class-Hiding Poisoning Statistics ===")
    print(f"Dataset: {input_csv}")
    print(f"Target class: {target_class}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Target class samples: {stats['target_samples']}")
    print(f"Poisoned samples: {stats['poisoned_samples']} ({poison_rate*100:.1f}%)")
    print(f"\nOriginal distribution: {original_counts}")
    print(f"Poisoned distribution: {new_counts}")
    print(f"\nPoisoned dataset saved to: {output_csv}")
    
    return stats


def create_poisoned_datasets(dataset_name, data_dir, poison_rates=[0.05, 0.10, 0.20], target_class=1):
    """
    Create multiple poisoned versions of a dataset with different poison rates.
    
    Args:
        dataset_name: Name of the dataset ('nusw', 'cic', 'cupid', 'cidds')
        data_dir: Path to the dataset directory
        poison_rates: List of poison rates to create (default: [0.05, 0.10, 0.20])
        target_class: The attack class to hide
    
    Returns:
        List of paths to created poisoned datasets
    """
    input_csv = os.path.join(data_dir, 'train.csv')
    
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Training file not found: {input_csv}")
    
    poisoned_files = []
    
    print(f"\n{'='*60}")
    print(f"Creating poisoned datasets for {dataset_name.upper()}")
    print(f"{'='*60}")
    
    for rate in poison_rates:
        output_csv = os.path.join(data_dir, f'train_poisoned_0_{int(rate*100):02d}.csv')
        
        print(f"\n--- Creating poisoned dataset with rate {rate} ---")
        stats = apply_class_hiding_poisoning(input_csv, output_csv, target_class, poison_rate=rate)
        poisoned_files.append(output_csv)
    
    print(f"\n{'='*60}")
    print(f"Created {len(poisoned_files)} poisoned datasets")
    print(f"{'='*60}\n")
    
    return poisoned_files


# Example usage
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Apply class-hiding poisoning to NIDS datasets')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['nusw', 'cic', 'cupid', 'cidds'],
                        help='Dataset to poison')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to data directory (default: data_real/{dataset}/)')
    parser.add_argument('--target_class', type=int, default=1,
                        help='Attack class to hide (default: 1)')
    parser.add_argument('--poison_rates', type=float, nargs='+', default=[0.05, 0.10, 0.20],
                        help='Poison rates to apply (default: 0.05 0.10 0.20)')
    
    args = parser.parse_args()
    
    # Set default data directory if not provided
    if args.data_dir is None:
        args.data_dir = f'data_real/{args.dataset}/'
    
    # Create poisoned datasets
    create_poisoned_datasets(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        poison_rates=args.poison_rates,
        target_class=args.target_class
    )
