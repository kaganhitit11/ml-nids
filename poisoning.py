import pandas as pd
import numpy as np
import os


# Simple feature predicates for each dataset
# Format: {dataset: {predicate_name: [(column, operator, value), ...]}}
PREDICATES = {
    'nusw': {
        'tcp_short': [('proto', '==', 'tcp'), ('dur', '<', 5.0)],
        'http_service': [('service', '==', 'http')],
        'high_rate': [('rate', '>', 1000.0)]
    },
    'cic': {
        'short_high_packets': [('Flow Duration', '<', 1000000), ('Total Fwd Packets', '>', 10)],
        'syn_pattern': [('SYN Flag Count', '>', 5)],
        'large_packets': [('Average Packet Size', '>', 1000)]
    },
    'cupid': {
        'short_high_volume': [('Flow Duration', '<', 1000000), ('Total Fwd Packet', '>', 10)],
        'syn_pattern': [('SYN Flag Count', '>', 5)]
    },
    'cidds': {
        'tcp_short': [('proto', '==', 'tcp'), ('duration', '<', 10.0)],
        'high_packets': [('packets', '>', 10)]
    }
}


def check_predicate(df, column, operator, value):
    """Check if a predicate matches for rows in dataframe."""
    if column not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    
    if operator == '==':
        # For categorical, convert to string for comparison
        return df[column].astype(str).str.lower() == str(value).lower()
    elif operator == '>':
        return df[column] > value
    elif operator == '<':
        return df[column] < value
    else:
        raise ValueError(f"Unsupported operator: {operator}")


def apply_feature_predicate_poisoning(input_csv, output_csv, dataset_name, 
                                     predicate_name, poison_rate=0.05):
    """
    Apply feature-predicate poisoning: Flip labels for samples matching feature predicates.
    
    Args:
        input_csv: Path to the input CSV file
        output_csv: Path to save the poisoned CSV file
        dataset_name: Name of the dataset ('nusw', 'cic', 'cupid', 'cidds')
        predicate_name: Name of the predicate to use
        poison_rate: Fraction of matching samples to poison (default: 0.05)
    
    Returns:
        dict with statistics about the poisoning
    """
    # Load dataset
    df = pd.read_csv(input_csv)
    
    # Detect label column
    label_col = 'Label' if 'Label' in df.columns else 'label'
    
    # Get predicates
    if dataset_name not in PREDICATES or predicate_name not in PREDICATES[dataset_name]:
        raise ValueError(f"Unknown predicate '{predicate_name}' for dataset '{dataset_name}'")
    
    predicates = PREDICATES[dataset_name][predicate_name]
    
    # Apply predicates (AND logic)
    mask = pd.Series([True] * len(df), index=df.index)
    for column, operator, value in predicates:
        predicate_mask = check_predicate(df, column, operator, value)
        mask &= predicate_mask
    
    # Only poison attack samples (not benign)
    mask &= (df[label_col] != 0)
    
    matching_indices = df[mask].index.tolist()
    
    if len(matching_indices) == 0:
        print(f"Warning: No samples match the predicates")
        return {
            'total_samples': len(df),
            'matching_samples': 0,
            'poisoned_samples': 0
        }
    
    # Get original distribution
    original_counts = df[label_col].value_counts().to_dict()
    
    # Select samples to poison
    num_to_poison = int(len(matching_indices) * poison_rate)
    np.random.seed(42)
    poisoned_indices = np.random.choice(matching_indices, size=num_to_poison, replace=False)
    
    # Flip labels to benign (0)
    df_poisoned = df.copy()
    df_poisoned.loc[poisoned_indices, label_col] = 0
    
    # Get new distribution
    new_counts = df_poisoned[label_col].value_counts().to_dict()
    
    # Save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_poisoned.to_csv(output_csv, index=False)
    
    # Statistics
    stats = {
        'total_samples': len(df),
        'matching_samples': len(matching_indices),
        'poisoned_samples': num_to_poison,
        'poison_rate': poison_rate,
        'predicate': predicates,
        'original_distribution': original_counts,
        'poisoned_distribution': new_counts
    }
    
    print(f"\n=== Feature-Predicate Poisoning Statistics ===")
    print(f"Dataset: {input_csv}")
    print(f"Predicate: {predicate_name} - {predicates}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Matching samples: {stats['matching_samples']}")
    print(f"Poisoned samples: {stats['poisoned_samples']} ({poison_rate*100:.1f}%)")
    print(f"\nOriginal distribution: {original_counts}")
    print(f"Poisoned distribution: {new_counts}")
    print(f"\nPoisoned dataset saved to: {output_csv}")
    
    return stats


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


def create_poisoned_datasets(dataset_name, data_dir, strategy='class_hiding', 
                           poison_rates=[0.05, 0.10, 0.20], target_class=1):
    """
    Create multiple poisoned versions of a dataset with different poison rates.
    
    Args:
        dataset_name: Name of the dataset ('nusw', 'cic', 'cupid', 'cidds')
        data_dir: Path to the dataset directory
        strategy: Poisoning strategy ('class_hiding' or 'feature_predicate')
        poison_rates: List of poison rates to create (default: [0.05, 0.10, 0.20])
        target_class: The attack class to hide (for class_hiding strategy)
    
    Returns:
        List of paths to created poisoned datasets
    """
    input_csv = os.path.join(data_dir, 'train.csv')
    
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Training file not found: {input_csv}")
    
    poisoned_files = []
    
    print(f"\n{'='*60}")
    print(f"Creating poisoned datasets for {dataset_name.upper()}")
    print(f"Strategy: {strategy}")
    print(f"{'='*60}")
    
    if strategy == 'class_hiding':
        for rate in poison_rates:
            output_csv = os.path.join(data_dir, f'train_poisoned_0_{int(rate*100):02d}.csv')
            print(f"\n--- Creating poisoned dataset with rate {rate} ---")
            stats = apply_class_hiding_poisoning(input_csv, output_csv, target_class, poison_rate=rate)
            poisoned_files.append(output_csv)
    
    else:  # feature_predicate - use all predicates for this dataset
        if dataset_name not in PREDICATES:
            raise ValueError(f"No predicates defined for dataset '{dataset_name}'")
        
        predicate_names = list(PREDICATES[dataset_name].keys())
        print(f"Using predicates: {predicate_names}")
        
        for predicate_name in predicate_names:
            print(f"\n{'='*60}")
            print(f"Predicate: {predicate_name}")
            print(f"{'='*60}")
            
            for rate in poison_rates:
                output_csv = os.path.join(data_dir, f'train_poisoned_{predicate_name}_0_{int(rate*100):02d}.csv')
                print(f"\n--- Creating poisoned dataset with rate {rate} ---")
                stats = apply_feature_predicate_poisoning(input_csv, output_csv, dataset_name, 
                                                         predicate_name, poison_rate=rate)
                poisoned_files.append(output_csv)
    
    print(f"\n{'='*60}")
    print(f"Created {len(poisoned_files)} poisoned datasets")
    print(f"{'='*60}\n")
    
    return poisoned_files


# Example usage
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Apply poisoning strategies to NIDS datasets')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['nusw', 'cic', 'cupid', 'cidds'],
                        help='Dataset to poison')
    parser.add_argument('--strategy', type=str, default='class_hiding',
                        choices=['class_hiding', 'feature_predicate'],
                        help='Poisoning strategy (default: class_hiding)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to data directory (default: data_real/{dataset}/)')
    parser.add_argument('--target_class', type=int, default=1,
                        help='Attack class to hide (for class_hiding, default: 1)')
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
        strategy=args.strategy,
        poison_rates=args.poison_rates,
        target_class=args.target_class
    )
