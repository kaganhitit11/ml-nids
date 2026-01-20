import pandas as pd
import numpy as np
import os
import warnings

# Try to use faster CSV engine if available
try:
    import pyarrow
    CSV_ENGINE = 'pyarrow'
except ImportError:
    CSV_ENGINE = 'c'
    warnings.warn("pyarrow not available, using default CSV engine. Install pyarrow for faster I/O: pip install pyarrow")

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
    """Check if a predicate matches for rows in dataframe (OPTIMIZED)."""
    if column not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    
    # Optimize by avoiding unnecessary type conversions
    col_data = df[column]
    
    if operator == '==':
        # Only convert to string if necessary
        if isinstance(value, str):
            return col_data.astype(str).str.lower() == value.lower()
        else:
            return col_data == value
    elif operator == '>':
        return col_data > value
    elif operator == '<':
        return col_data < value
    else:
        raise ValueError(f"Unsupported operator: {operator}")

def flip_label_bidirectional(df, indices, label_col):
    """
    Flip labels bidirectionally for given indices (VECTORIZED VERSION).
    - Attack (1 or non-'Benign') -> Benign (0 or 'Benign')
    - Benign (0 or 'Benign') -> Attack (1 or 'Attack')
    
    Returns counts of flips in each direction.
    """
    # Create a shallow copy and only modify the label column
    df_poisoned = df.copy()
    
    # Convert indices to numpy array for faster operations
    indices = np.array(indices)
    
    if pd.api.types.is_numeric_dtype(df[label_col]):
        # Numeric labels: 0 = benign, 1 = attack
        # Get current labels for selected indices
        current_labels = df.loc[indices, label_col].values
        
        # Create flipped labels vectorially
        flipped_labels = 1 - current_labels  # 0->1, 1->0
        
        # Count flips
        benign_to_attack = (current_labels == 0).sum()
        attack_to_benign = (current_labels == 1).sum()
        
        # Apply flips all at once
        df_poisoned.loc[indices, label_col] = flipped_labels
    else:
        # String labels - vectorized approach
        current_labels = df.loc[indices, label_col].astype(str).str.lower().str.strip().values
        
        # Create mask for benign samples
        is_benign = np.isin(current_labels, ['0', 'benign'])
        
        # Count flips
        benign_to_attack = is_benign.sum()
        attack_to_benign = (~is_benign).sum()
        
        # Create flipped labels array
        flipped_labels = np.where(is_benign, 'Attack', 'Benign')
        
        # Apply flips all at once
        df_poisoned.loc[indices, label_col] = flipped_labels
    
    return df_poisoned, attack_to_benign, benign_to_attack

def apply_feature_predicate_poisoning(input_csv, output_csv, dataset_name, poison_rate=0.05):
    """
    Apply feature-predicate poisoning with global budget and bidirectional flipping.
    Prioritizes samples matching predicates, but uses global budget.
    """
    print(f"Reading dataset from {input_csv}...")
    df = pd.read_csv(input_csv, engine=CSV_ENGINE)
    # Robust label finding
    label_col = next((c for c in df.columns if c.lower() == 'label'), None)
    if not label_col: 
        raise ValueError("Label column not found")
    
    if dataset_name not in PREDICATES:
        raise ValueError(f"No predicates defined for dataset '{dataset_name}'")
        
    # Start with a mask of all False (using numpy for speed)
    mask = np.zeros(len(df), dtype=bool)
    
    print(f"Applying COMBINED predicates for {dataset_name}...")
    
    # Iterate over ALL predicates for this dataset and UNION them
    for p_name, rules in PREDICATES[dataset_name].items():
        rule_mask = np.ones(len(df), dtype=bool)
        for column, operator, value in rules:
            predicate_result = check_predicate(df, column, operator, value)
            rule_mask &= predicate_result.values
        
        # Combine: If it matches this rule OR previous rules
        mask |= rule_mask
    
    # Convert to list once at the end (more efficient than df[mask].index.tolist())
    matching_indices = np.where(mask)[0].tolist()
    
    # Global budget calculation
    num_to_poison = int(len(df) * poison_rate)
    if num_to_poison == 0:
        print(f"No samples to poison (rate too low)")
        return {'total_samples': len(df), 'poisoned_samples': 0}
    
    # Prioritize matching indices, expand pool if necessary
    if len(matching_indices) >= num_to_poison:
        # Enough matching candidates
        np.random.seed(42)
        poisoned_indices = np.random.choice(matching_indices, size=num_to_poison, replace=False).tolist()
    else:
        # Need to expand pool with non-matching samples
        non_matching_indices = np.where(~mask)[0]  # More efficient than df[~mask].index.tolist()
        np.random.seed(42)
        additional_needed = num_to_poison - len(matching_indices)
        additional_indices = np.random.choice(non_matching_indices, size=additional_needed, replace=False)
        poisoned_indices = matching_indices + additional_indices.tolist()
    
    # Apply bidirectional label flip
    df_poisoned, attack_to_benign, benign_to_attack = flip_label_bidirectional(df, poisoned_indices, label_col)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_poisoned.to_csv(output_csv, index=False)
    
    print(f"\n=== Feature-Predicate Poisoning Statistics (Global Budget) ===")
    print(f"Dataset: {input_csv}")
    print(f"Total samples: {len(df)}")
    print(f"Matching candidates: {len(matching_indices)}")
    print(f"Global poison budget: {num_to_poison} ({poison_rate*100:.1f}%)")
    print(f"Attack→Benign flips: {attack_to_benign}")
    print(f"Benign→Attack flips: {benign_to_attack}")
    print(f"Original distribution: {df[label_col].value_counts().to_dict()}")
    print(f"Poisoned distribution: {df_poisoned[label_col].value_counts().to_dict()}")
    
    return {'total_samples': len(df), 'poisoned_samples': num_to_poison, 
            'attack_to_benign': attack_to_benign, 'benign_to_attack': benign_to_attack}


def apply_class_hiding_poisoning(input_csv, output_csv, target_class, poison_rate=0.05):
    """Apply class-hiding poisoning with global budget and bidirectional flipping."""
    print(f"Reading dataset from {input_csv}...")
    df = pd.read_csv(input_csv, engine=CSV_ENGINE)
    label_col = 'Label' if 'Label' in df.columns else 'label'
    
    target_mask = (df[label_col] == target_class).values  # Convert to numpy array
    target_indices = np.where(target_mask)[0]  # More efficient than df[target_mask].index.tolist()
    
    # Global budget calculation
    num_to_poison = int(len(df) * poison_rate)
    if num_to_poison == 0:
        return {'total_samples': len(df), 'poisoned_samples': 0}
    
    # Prioritize target class, expand pool if necessary
    if len(target_indices) >= num_to_poison:
        # Enough target class samples
        np.random.seed(42)
        poisoned_indices = np.random.choice(target_indices, size=num_to_poison, replace=False).tolist()
    else:
        # Need to expand pool with other class samples
        non_target_indices = np.where(~target_mask)[0]  # More efficient
        np.random.seed(42)
        additional_needed = num_to_poison - len(target_indices)
        if additional_needed > len(non_target_indices):
            additional_needed = len(non_target_indices)
        additional_indices = np.random.choice(non_target_indices, size=additional_needed, replace=False)
        poisoned_indices = target_indices.tolist() + additional_indices.tolist()
    
    # Apply bidirectional label flip
    df_poisoned, attack_to_benign, benign_to_attack = flip_label_bidirectional(df, poisoned_indices, label_col)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_poisoned.to_csv(output_csv, index=False)
    
    print(f"\n=== Class-Hiding Poisoning Statistics (Global Budget) ===")
    print(f"Target class: {target_class}")
    print(f"Total samples: {len(df)}")
    print(f"Target class samples: {len(target_indices)}")
    print(f"Global poison budget: {num_to_poison} ({poison_rate*100:.1f}%)")
    print(f"Attack→Benign flips: {attack_to_benign}")
    print(f"Benign→Attack flips: {benign_to_attack}")
    print(f"Original distribution: {df[label_col].value_counts().to_dict()}")
    print(f"Poisoned distribution: {df_poisoned[label_col].value_counts().to_dict()}")
    
    return {'total_samples': len(df), 'poisoned_samples': num_to_poison,
            'attack_to_benign': attack_to_benign, 'benign_to_attack': benign_to_attack}


def apply_disagreement_poisoning(input_csv, output_csv, conf_csv1, conf_csv2, poison_rate=0.05):
    """Apply disagreement-based poisoning with global budget and bidirectional flipping."""
    print(f"Reading dataset from {input_csv}...")
    df = pd.read_csv(input_csv, engine=CSV_ENGINE)
    label_col = 'Label' if 'Label' in df.columns else 'label'

    print(f"Reading confidence files...")
    c1 = pd.read_csv(conf_csv1, engine=CSV_ENGINE)
    c2 = pd.read_csv(conf_csv2, engine=CSV_ENGINE)
    c1 = c1[c1['epoch'] == c1['epoch'].max()].sort_values('sample_idx').reset_index(drop=True)
    c2 = c2[c2['epoch'] == c2['epoch'].max()].sort_values('sample_idx').reset_index(drop=True)

    # Remove attack-only filter - consider ALL samples with disagreement
    candidates_mask = (c1['predicted_label'] != c2['predicted_label']).values
    candidate_indices = np.where(candidates_mask)[0]
    
    # Global budget calculation
    num_to_poison = int(len(df) * poison_rate)
    if num_to_poison == 0:
        return {'total_samples': len(df), 'poisoned_samples': 0}
    
    # Prioritize disagreement candidates, expand pool if necessary
    if len(candidate_indices) >= num_to_poison:
        # Enough disagreement candidates - sort by confidence difference
        conf_diff = np.abs(c1.loc[candidate_indices, 'confidence'].values - c2.loc[candidate_indices, 'confidence'].values)
        sorted_idx = np.argsort(conf_diff)[::-1]  # Sort descending
        sorted_candidates = candidate_indices[sorted_idx]
        poisoned_indices = sorted_candidates[:num_to_poison].tolist()
    else:
        # Need to expand pool with non-disagreement samples
        non_candidate_indices = np.where(~candidates_mask)[0]
        np.random.seed(42)
        additional_needed = num_to_poison - len(candidate_indices)
        additional_indices = np.random.choice(non_candidate_indices, size=additional_needed, replace=False)
        poisoned_indices = candidate_indices.tolist() + additional_indices.tolist()

    # Apply bidirectional label flip
    df_poisoned, attack_to_benign, benign_to_attack = flip_label_bidirectional(df, poisoned_indices, label_col)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_poisoned.to_csv(output_csv, index=False)

    print(f"\n=== Disagreement-Based Poisoning Statistics (Global Budget) ===")
    print(f"Total samples: {len(df)}")
    print(f"Disagreement candidates: {len(candidate_indices)}")
    print(f"Global poison budget: {num_to_poison} ({poison_rate*100:.1f}%)")
    print(f"Attack→Benign flips: {attack_to_benign}")
    print(f"Benign→Attack flips: {benign_to_attack}")
    print(f"Original distribution: {df[label_col].value_counts().to_dict()}")
    print(f"Poisoned distribution: {df_poisoned[label_col].value_counts().to_dict()}")

    return {'total_samples': len(df), 'poisoned_samples': num_to_poison,
            'attack_to_benign': attack_to_benign, 'benign_to_attack': benign_to_attack}


def apply_confidence_based_poisoning(input_csv, output_csv, confidence_csv, poison_rate=0.05):
    """Apply confidence-based poisoning with global budget and bidirectional flipping."""
    print(f"Reading dataset from {input_csv}...")
    df = pd.read_csv(input_csv, engine=CSV_ENGINE)
    label_col = 'Label' if 'Label' in df.columns else 'label'
    
    print(f"Reading confidence file...")
    confidence_df = pd.read_csv(confidence_csv, engine=CSV_ENGINE)
    last_epoch = confidence_df['epoch'].max()
    confidence_df = confidence_df[confidence_df['epoch'] == last_epoch]
    confidence_df = confidence_df.sort_values('sample_idx').reset_index(drop=True)
    
    # Global budget calculation
    num_to_poison = int(len(df) * poison_rate)
    if num_to_poison == 0:
        return {'total_samples': len(df), 'poisoned_samples': 0}
    
    # Sort ALL samples by confidence (lowest first)
    all_confidences = confidence_df['confidence'].values
    sorted_indices = np.argsort(all_confidences)
    lowest_conf_indices = sorted_indices[:num_to_poison]
    
    # Apply bidirectional label flip
    df_poisoned, attack_to_benign, benign_to_attack = flip_label_bidirectional(df, lowest_conf_indices, label_col)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_poisoned.to_csv(output_csv, index=False)
    
    print(f"\n=== Confidence-Based Poisoning Statistics (Global Budget) ===")
    print(f"Total samples: {len(df)}")
    print(f"Global poison budget: {num_to_poison} ({poison_rate*100:.1f}%)")
    print(f"Attack→Benign flips: {attack_to_benign}")
    print(f"Benign→Attack flips: {benign_to_attack}")
    print(f"Original distribution: {df[label_col].value_counts().to_dict()}")
    print(f"Poisoned distribution: {df_poisoned[label_col].value_counts().to_dict()}")
    
    return {'total_samples': len(df), 'poisoned_samples': num_to_poison,
            'attack_to_benign': attack_to_benign, 'benign_to_attack': benign_to_attack}


def apply_temporal_window_poisoning(input_csv, output_csv, poison_rate=0.05, window_freq='5min'):
    """
    Apply Pointwise anomaly-in-time poisoning (CUPID only).
    OPTIMIZED VERSION: Uses vectorized operations instead of slow loops.
    """
    print(f"Loading {input_csv} for Temporal Anomaly Poisoning...")
    df = pd.read_csv(input_csv, engine=CSV_ENGINE)
    
    # --- FIX 1: Clean column names (removes spaces like ' Timestamp') ---
    df.columns = df.columns.str.strip()
    
    label_col = 'Label' if 'Label' in df.columns else 'label'

    # Check for Timestamp and required features
    if 'Timestamp' not in df.columns:
        print(f"Available columns: {df.columns.tolist()}") # Debug info
        raise ValueError("Dataset missing 'Timestamp' column required for temporal poisoning.")
    
    missing_features = [f for f in TEMPORAL_ANOMALY_FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Dataset missing required features: {missing_features}")

    # 1. Assign windows
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %I:%M:%S %p', errors='coerce')
    
    # Drop invalid timestamps
    if df['Timestamp'].isna().any():
        print(f"Warning: Dropped {df['Timestamp'].isna().sum()} rows with invalid timestamps.")
        df = df.dropna(subset=['Timestamp'])
    
    # Create a column for the window start time (bucket)
    df['window_start'] = df['Timestamp'].dt.floor(window_freq)

    # --- FIX 2: Vectorized Calculation (No more 'iterrows' loop) ---
    print("Computing window statistics (Vectorized)...")
    
    # Calculate Mean and Std for every window for all features at once
    # Result is indexed by 'window_start'
    window_stats = df.groupby('window_start')[TEMPORAL_ANOMALY_FEATURES].agg(['mean', 'std'])
    
    # Flatten MultiIndex columns (e.g., ('Flow Duration', 'mean') -> 'Flow Duration_mean')
    window_stats.columns = ['_'.join(col).strip() for col in window_stats.columns.values]
    
    # Merge stats back to the original dataframe based on window_start
    # This aligns the window mean/std with every single sample row instantly
    df_merged = df.merge(window_stats, on='window_start', how='left')
    
    # Calculate Deviation Score Vectorially
    epsilon = 1e-9
    df_merged['deviation_score'] = 0.0
    
    for feature in TEMPORAL_ANOMALY_FEATURES:
        mu = df_merged[f'{feature}_mean']
        sigma = df_merged[f'{feature}_std'].fillna(0)
        
        # Vectorized Z-score: abs(x - mu) / sigma
        z_scores = ((df_merged[feature] - mu) / (sigma + epsilon)).abs()
        df_merged['deviation_score'] += z_scores

    # 4. Rank samples globally
    # Get indices of top K samples with highest deviation scores
    N = len(df)
    K = int(N * poison_rate)
    
    print(f"Ranking samples... (Targeting top {K})")
    sorted_indices = df_merged.sort_values(by='deviation_score', ascending=False).index
    samples_to_poison = sorted_indices[:K].tolist()
    
    # 5. Apply Poisoning
    # Note: We pass 'df' (the original), but we use indices from 'df_merged' 
    # (which are preserved/aligned if we didn't reset index destructively)
    df_poisoned, attack_to_benign, benign_to_attack = flip_label_bidirectional(df, samples_to_poison, label_col)
    
    # Save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_poisoned.to_csv(output_csv, index=False)

    # Statistics
    stats = {
        'total_samples': N,
        'global_poison_budget': K,
        'attack_to_benign': attack_to_benign,
        'benign_to_attack': benign_to_attack,
        'original_distribution': df[label_col].value_counts().to_dict(),
        'poisoned_distribution': df_poisoned[label_col].value_counts().to_dict()
    }

    print(f"\n=== Pointwise Anomaly-in-Time Poisoning Statistics (Vectorized) ===")
    print(f"Window Frequency: {window_freq}")
    print(f"Total samples: {N}")
    print(f"Global poison budget: {K} ({poison_rate*100:.1f}%)")
    print(f"Attack→Benign flips: {attack_to_benign}")
    print(f"Benign→Attack flips: {benign_to_attack}")
    print(f"Original distribution: {stats['original_distribution']}")
    print(f"Poisoned distribution: {stats['poisoned_distribution']}")
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
          <data_dir>/poisoned/<strategy>[_<extra>]/<PPP>/train.csv
        Example:
          data_real/nusw/poisoned/class_hiding/005/train.csv
          data_real/nusw/poisoned/feature_predicate_tcp_short/010/train.csv
        """
        # Build strategy directory name
        strategy_dir = strategy_name
        if extra:
            strategy_dir = f"{strategy_name}_{extra}"
        
        # Create full directory path: data_dir/poisoned/strategy/percentage/
        out_dir = os.path.join(data_dir, 'poisoned', strategy_dir, _pct_str(rate))
        os.makedirs(out_dir, exist_ok=True)
        
        # Always save as train.csv
        return os.path.join(out_dir, 'train.csv')
    
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
        confidence_csv = os.path.join('/home/ohitit20', log_dir, log_subdir, f"{log_subdir}_seed{seed}_per_sample_metrics.csv")
        
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
        conf_csv1 = os.path.join('/home/ohitit20', log_dir, log_subdir, f"{log_subdir}_seed{seed}_per_sample_metrics.csv")
        conf_csv2 = os.path.join('/home/ohitit20', log_dir, log_subdir, f"{log_subdir}_seed{seed2}_per_sample_metrics.csv")
        
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
            # Default window freq is 5min (5 minutes)
            apply_temporal_window_poisoning(input_csv, output_csv, poison_rate=rate, window_freq='5min')
            poisoned_files.append(output_csv)

    elif strategy == 'feature_predicate':
        for rate in poison_rates:
            output_csv = _poisoned_output_path('feature_predicate', rate)
            print(f"\n--- Creating poisoned dataset with rate {rate} (Combined Predicates) ---")
            
            apply_feature_predicate_poisoning(input_csv, output_csv, dataset_name, poison_rate=rate)
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