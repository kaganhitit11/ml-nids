import pandas as pd
import numpy as np
import os

# Simple feature predicates for each dataset
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

# Feature set for temporal anomaly poisoning
TEMPORAL_ANOMALY_FEATURES = [
    'Flow Duration',
    'Total Fwd Packet',
    'Total Bwd packets',
    'Total Length of Fwd Packet',
    'Total Length of Bwd Packet'
]

def check_predicate(df, column, operator, value):
    """Check if a predicate matches for rows in dataframe."""
    if column not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    
    if operator == '==':
        return df[column].astype(str).str.lower() == str(value).lower()
    elif operator == '>':
        return df[column] > value
    elif operator == '<':
        return df[column] < value
    else:
        raise ValueError(f"Unsupported operator: {operator}")


def apply_feature_predicate_poisoning(input_csv, output_csv, dataset_name, 
                                     predicate_name, poison_rate=0.05):
    """Apply feature-predicate poisoning."""
    df = pd.read_csv(input_csv)
    label_col = 'Label' if 'Label' in df.columns else 'label'
    
    if dataset_name not in PREDICATES or predicate_name not in PREDICATES[dataset_name]:
        raise ValueError(f"Unknown predicate '{predicate_name}' for dataset '{dataset_name}'")
    
    predicates = PREDICATES[dataset_name][predicate_name]
    mask = pd.Series([True] * len(df), index=df.index)
    for column, operator, value in predicates:
        mask &= check_predicate(df, column, operator, value)
    
    # Only poison attack samples
    mask &= (df[label_col] != 0)
    matching_indices = df[mask].index.tolist()
    
    if len(matching_indices) == 0:
        return {'total_samples': len(df), 'poisoned_samples': 0}
    
    original_counts = df[label_col].value_counts().to_dict()
    num_to_poison = int(len(matching_indices) * poison_rate)
    
    np.random.seed(42)
    poisoned_indices = np.random.choice(matching_indices, size=num_to_poison, replace=False)
    
    df_poisoned = df.copy()
    df_poisoned.loc[poisoned_indices, label_col] = 0
    
    new_counts = df_poisoned[label_col].value_counts().to_dict()
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_poisoned.to_csv(output_csv, index=False)
    
    print(f"\n=== Feature-Predicate Poisoning Statistics ===")
    print(f"Dataset: {input_csv} | Predicate: {predicate_name}")
    print(f"Poisoned samples: {num_to_poison} ({poison_rate*100:.1f}%)")
    
    return {'total_samples': len(df), 'poisoned_samples': num_to_poison}


def apply_class_hiding_poisoning(input_csv, output_csv, target_class, poison_rate=0.05):
    """Apply class-hiding poisoning."""
    df = pd.read_csv(input_csv)
    label_col = 'Label' if 'Label' in df.columns else 'label'
    
    target_mask = df[label_col] == target_class
    target_indices = df[target_mask].index.tolist()
    
    if len(target_indices) == 0:
        return {'total_samples': len(df), 'poisoned_samples': 0}
    
    num_to_poison = int(len(target_indices) * poison_rate)
    np.random.seed(42)
    poisoned_indices = np.random.choice(target_indices, size=num_to_poison, replace=False)
    
    df_poisoned = df.copy()
    df_poisoned.loc[poisoned_indices, label_col] = 0
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_poisoned.to_csv(output_csv, index=False)
    
    print(f"\n=== Class-Hiding Poisoning Statistics ===")
    print(f"Target class: {target_class} | Poisoned samples: {num_to_poison}")
    
    return {'total_samples': len(df), 'poisoned_samples': num_to_poison}


def apply_disagreement_poisoning(input_csv, output_csv, conf_csv1, conf_csv2, poison_rate=0.05):
    """Apply disagreement-based poisoning."""
    df = pd.read_csv(input_csv)
    label_col = 'Label' if 'Label' in df.columns else 'label'

    c1 = pd.read_csv(conf_csv1)
    c2 = pd.read_csv(conf_csv2)
    c1 = c1[c1['epoch'] == c1['epoch'].max()].sort_values('sample_idx').reset_index(drop=True)
    c2 = c2[c2['epoch'] == c2['epoch'].max()].sort_values('sample_idx').reset_index(drop=True)

    candidates_mask = (df[label_col] != 0) & (c1['predicted_label'] != c2['predicted_label'])
    candidate_indices = df[candidates_mask].index.tolist()
    
    if len(candidate_indices) == 0:
        return {'total_samples': len(df), 'poisoned_samples': 0}

    conf_diff = (c1.loc[candidate_indices, 'confidence'] - c2.loc[candidate_indices, 'confidence']).abs()
    sorted_candidates = conf_diff.sort_values(ascending=False).index.tolist()

    num_to_poison = int(len(df) * poison_rate)
    if num_to_poison > len(sorted_candidates):
        num_to_poison = len(sorted_candidates)
    
    poisoned_indices = sorted_candidates[:num_to_poison]

    df_poisoned = df.copy()
    df_poisoned.loc[poisoned_indices, label_col] = 0

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_poisoned.to_csv(output_csv, index=False)

    print(f"\n=== Disagreement-Based Poisoning Statistics ===")
    print(f"Actually poisoned: {num_to_poison}")

    return {'total_samples': len(df), 'poisoned_samples': num_to_poison}


def apply_confidence_based_poisoning(input_csv, output_csv, confidence_csv, poison_rate=0.05):
    """Apply confidence-based poisoning."""
    df = pd.read_csv(input_csv)
    label_col = 'Label' if 'Label' in df.columns else 'label'
    
    confidence_df = pd.read_csv(confidence_csv)
    last_epoch = confidence_df['epoch'].max()
    confidence_df = confidence_df[confidence_df['epoch'] == last_epoch]
    confidence_df = confidence_df.sort_values('sample_idx').reset_index(drop=True)
    
    attack_mask = df[label_col] != 0
    attack_indices = df[attack_mask].index.tolist()
    
    if len(attack_indices) == 0:
        return {'total_samples': len(df), 'poisoned_samples': 0}
    
    attack_confidences = confidence_df.loc[attack_indices, 'confidence'].values
    sorted_indices = np.argsort(attack_confidences)
    lowest_conf_indices = np.array(attack_indices)[sorted_indices]
    
    num_to_poison = int(len(attack_indices) * poison_rate)
    poisoned_indices = lowest_conf_indices[:num_to_poison]
    
    df_poisoned = df.copy()
    df_poisoned.loc[poisoned_indices, label_col] = 0
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_poisoned.to_csv(output_csv, index=False)
    
    print(f"\n=== Confidence-Based Poisoning Statistics ===")
    print(f"Poisoned samples: {num_to_poison}")
    
    return {'total_samples': len(df), 'poisoned_samples': num_to_poison}


def apply_temporal_window_poisoning(input_csv, output_csv, poison_rate=0.05, window_freq='5T'):
    """
    Apply Pointwise anomaly-in-time poisoning (CUPID only).

    Algorithm:
    1. Assign windows.
    2. For each window w, compute baseline stats mu_w, sigma_w for a feature set F.
    3. For each sample i in window w, compute a deviation score A_i.
    4. Rank all samples globally by A_i.
    5. Poison top K = floor(pN) samples.

    This method poisons the most "temporally unusual" points globally.
    
    Args:
        input_csv: Path to input CSV.
        output_csv: Path to output CSV.
        poison_rate: Fraction of total data to poison (p).
        window_freq: Pandas offset alias for window size (default '5T' = 5 mins).
    """
    print(f"Loading {input_csv} for Temporal Anomaly Poisoning...")
    df = pd.read_csv(input_csv)
    label_col = 'Label' if 'Label' in df.columns else 'label'

    # Check for Timestamp and required features
    if 'Timestamp' not in df.columns:
        raise ValueError("Dataset missing 'Timestamp' column required for temporal poisoning.")
    
    missing_features = [f for f in TEMPORAL_ANOMALY_FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Dataset missing required features for temporal anomaly poisoning: {missing_features}")

    # 1. Assign windows (Parse, Sort, and Bin)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    if df['Timestamp'].isna().any():
        print(f"Warning: Dropped {df['Timestamp'].isna().sum()} rows with invalid timestamps.")
        df = df.dropna(subset=['Timestamp'])
    
    # Sort and set index for grouping
    df = df.sort_values(by='Timestamp')
    df_indexed = df.set_index('Timestamp')

    # 2. Compute window baseline stats for feature set F
    # Group by window freq and calculate mean and std for the feature set
    window_stats = df_indexed.groupby(pd.Grouper(freq=window_freq))[TEMPORAL_ANOMALY_FEATURES].agg(['mean', 'std'])
    
    # 3. Compute a deviation score for each sample i
    epsilon = 1e-9  # Small constant for numerical stability
    deviation_scores = []
    
    # Iterate through each sample, find its window's stats, and compute the score
    for idx, row in df_indexed.iterrows():
        # Find the window for the current sample
        window_start = idx.floor(window_freq)
        
        try:
            stats = window_stats.loc[window_start]
        except KeyError:
            # This should not happen if binning is correct, but for safety
            deviation_scores.append(0)
            continue

        score = 0
        for feature in TEMPORAL_ANOMALY_FEATURES:
            mu_w = stats[feature]['mean']
            sigma_w = stats[feature]['std']
            x_i = row[feature]
            
            # Handle cases where std is NaN (e.g., window has only 1 sample)
            if np.isnan(sigma_w):
                 sigma_w = 0
            
            # Compute z-score component
            z_score = abs((x_i - mu_w) / (sigma_w + epsilon))
            score += z_score
            
        deviation_scores.append(score)

    # Add scores to the original dataframe (which is sorted by time)
    df['deviation_score'] = deviation_scores
    
    # 4. Rank all samples globally by A_i (deviation_score)
    # We only want to poison Attack samples (Label=1), so we prioritize them
    # We will sort by deviation score descending, but we need a way to only pick attacks.
    # A simple way is to get the indices of the top K scores, and then from those, filter for attacks.
    
    # Get indices of all samples sorted by deviation score descending
    sorted_indices = df.sort_values(by='deviation_score', ascending=False).index
    
    # 5. Poison top K = floor(pN) samples.
    # We need to find the top K *attack* samples based on deviation score.
    
    N = len(df)
    K = int(N * poison_rate) #
    
    # Filter sorted indices to keep only those that are originally attacks
    attack_indices_sorted = [idx for idx in sorted_indices if df.loc[idx, label_col] == 1]
    
    # Select the top K from the sorted attack indices
    samples_to_poison = attack_indices_sorted[:K]
    
    # Apply Poisoning (Targeted: Attack -> Benign)
    df_poisoned = df.copy()
    df_poisoned.loc[samples_to_poison, label_col] = 0
    
    # Drop the deviation_score column before saving
    df_poisoned = df_poisoned.drop(columns=['deviation_score'])

    # Save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_poisoned.to_csv(output_csv, index=False)

    total_poisoned = len(samples_to_poison)
    # Statistics
    stats = {
        'total_samples': N,
        'target_poison_count_K': K,
        'actual_attacks_flipped': total_poisoned,
        'original_distribution': df[label_col].value_counts().to_dict(),
        'poisoned_distribution': df_poisoned[label_col].value_counts().to_dict()
    }

    print(f"\n=== Pointwise Anomaly-in-Time Poisoning Statistics ===")
    print(f"Window Frequency: {window_freq}")
    print(f"Target poison count (K): {K} ({poison_rate*100:.1f}%)")
    print(f"Actual Attack samples masked (1->0): {total_poisoned}")
    print(f"Poisoned dataset saved to: {output_csv}")

    return stats


def create_poisoned_datasets(dataset_name, data_dir, strategy='class_hiding', 
                           poison_rates=[0.05, 0.10, 0.20], target_class=1,
                           model_type=None, seed=42, seed2=None, log_dir='train_logs'):
    """Create multiple poisoned versions of a dataset."""
    input_csv = os.path.join(data_dir, 'train.csv')
    
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Training file not found: {input_csv}")
    
    poisoned_files = []

    def _pct_str(rate: float) -> str:
        """Convert a fraction (e.g., 0.05) to a zero-padded percent string (e.g., '005')."""
        return f"{int(round(rate * 100)):03d}"

    def _poisoned_output_path(strategy_name: str, rate: float, extra: str | None = None) -> str:
        """
        Output path convention:
          <data_dir>/poisoned/train_poisoned_<strategy>[_<extra>]_<PPP>.csv
        Example:
          data_real/nusw/poisoned/train_poisoned_class_hiding_005.csv
        """
        out_dir = os.path.join(data_dir, 'poisoned')
        os.makedirs(out_dir, exist_ok=True)
        parts = ['train', 'poisoned', strategy_name]
        if extra:
            parts.append(extra)
        filename = f"{'_'.join(parts)}_{_pct_str(rate)}.csv"
        return os.path.join(out_dir, filename)
    
    print(f"\n{'='*60}")
    print(f"Creating poisoned datasets for {dataset_name.upper()}")
    print(f"Strategy: {strategy}")
    print(f"{'='*60}")
    
    if strategy == 'class_hiding':
        for rate in poison_rates:
            output_csv = _poisoned_output_path('class_hiding', rate)
            print(f"\n--- Creating poisoned dataset with rate {rate} ---")
            apply_class_hiding_poisoning(input_csv, output_csv, target_class, poison_rate=rate)
            poisoned_files.append(output_csv)
    
    elif strategy == 'confidence_based':
        if model_type is None:
            raise ValueError("model_type is required for confidence_based strategy")
        log_subdir = f"{dataset_name}_{model_type}"
        confidence_csv = os.path.join(log_dir, log_subdir, f"{log_subdir}_seed{seed}_per_sample_metrics.csv")
        
        if not os.path.exists(confidence_csv):
            raise FileNotFoundError(f"Confidence log file not found: {confidence_csv}")
            
        for rate in poison_rates:
            output_csv = _poisoned_output_path('confidence_based', rate)
            print(f"\n--- Creating poisoned dataset with rate {rate} ---")
            apply_confidence_based_poisoning(input_csv, output_csv, confidence_csv, poison_rate=rate)
            poisoned_files.append(output_csv)
            
    elif strategy == 'disagreement':
        if model_type is None or seed2 is None:
            raise ValueError("model_type and seed2 required for disagreement strategy")
        log_subdir = f"{dataset_name}_{model_type}"
        conf_csv1 = os.path.join(log_dir, log_subdir, f"{log_subdir}_seed{seed}_per_sample_metrics.csv")
        conf_csv2 = os.path.join(log_dir, log_subdir, f"{log_subdir}_seed{seed2}_per_sample_metrics.csv")
        
        if not os.path.exists(conf_csv1) or not os.path.exists(conf_csv2):
            raise FileNotFoundError("One or both confidence log files not found.")

        for rate in poison_rates:
            output_csv = _poisoned_output_path('disagreement', rate)
            print(f"\n--- Creating poisoned dataset with rate {rate} ---")
            apply_disagreement_poisoning(input_csv, output_csv, conf_csv1, conf_csv2, poison_rate=rate)
            poisoned_files.append(output_csv)

    elif strategy == 'temporal':
        # Temporal Poisoning (CUPID only check)
        if dataset_name.lower() != 'cupid':
            raise ValueError("Temporal poisoning is currently only supported for CUPID dataset (requires Timestamp).")
        
        for rate in poison_rates:
            output_csv = _poisoned_output_path('temporal', rate)
            print(f"\n--- Creating poisoned dataset with rate {rate} ---")
            # Default window freq is 5T (5 minutes)
            apply_temporal_window_poisoning(input_csv, output_csv, poison_rate=rate, window_freq='5T')
            poisoned_files.append(output_csv)

    elif strategy == 'feature_predicate':
        if dataset_name not in PREDICATES:
            raise ValueError(f"No predicates defined for dataset '{dataset_name}'")
        predicate_names = list(PREDICATES[dataset_name].keys())
        
        for predicate_name in predicate_names:
            for rate in poison_rates:
                output_csv = _poisoned_output_path('feature_predicate', rate, extra=predicate_name)
                print(f"\n--- Creating poisoned dataset with rate {rate} ---")
                apply_feature_predicate_poisoning(input_csv, output_csv, dataset_name, predicate_name, poison_rate=rate)
                poisoned_files.append(output_csv)
    
    else:
        raise ValueError(f"Invalid strategy: {strategy}")
    
    print(f"\n{'='*60}")
    print(f"Created {len(poisoned_files)} poisoned datasets")
    print(f"{'='*60}\n")
    
    return poisoned_files


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Apply poisoning strategies to NIDS datasets')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['nusw', 'cic', 'cupid', 'cidds'],
                        help='Dataset to poison')
    parser.add_argument('--strategy', type=str, default='class_hiding',
                        choices=['class_hiding', 'feature_predicate', 'confidence_based', 'disagreement', 'temporal'],
                        help='Poisoning strategy (default: class_hiding)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to data directory (default: data_real/{dataset}/)')
    parser.add_argument('--target_class', type=int, default=1,
                        help='Attack class to hide (for class_hiding, default: 1)')
    parser.add_argument('--poison_rates', type=float, nargs='+', default=[0.05, 0.10, 0.20],
                        help='Poison rates to apply (default: 0.05 0.10 0.20)')
    parser.add_argument('--model_type', type=str, default=None,
                        choices=['cnn', 'rnn', 'mlp', 'logistic', 'random_forest'],
                        help='Model type for confidence_based strategy')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed used in baseline training')
    parser.add_argument('--seed2', type=int, default=None,
                        help='Seed 2 (required for disagreement strategy)')
    parser.add_argument('--log_dir', type=str, default='train_logs',
                        help='Directory containing training logs')
    
    args = parser.parse_args()
    
    if args.data_dir is None:
        args.data_dir = f'data_real/{args.dataset}/'
    
    create_poisoned_datasets(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        strategy=args.strategy,
        poison_rates=args.poison_rates,
        target_class=args.target_class,
        model_type=args.model_type,
        seed=args.seed,
        seed2=args.seed2,
        log_dir=args.log_dir
    )