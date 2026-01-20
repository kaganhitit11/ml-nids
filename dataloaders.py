import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np


class UNSW_NB15_Dataset(Dataset):
    """Simple dataset class for UNSW_NB15"""
    
    def __init__(self, csv_path, scaler=None, label_encoders=None, fit_transform=False):
        """
        Args:
            csv_path: Path to the CSV file
            scaler: StandardScaler for numerical features (pass fitted scaler for test set)
            label_encoders: Dict of LabelEncoders for categorical features
            fit_transform: If True, fit the scalers (for training set)
        """
        # Load data
        self.df = pd.read_csv(csv_path)
        
        # Separate features and label
        self.labels = self.df['label'].values
        
        # Drop id, label, and attack_cat columns
        feature_cols = [col for col in self.df.columns if col not in ['id', 'label', 'attack_cat']]
        self.features_df = self.df[feature_cols].copy()
        
        # Identify categorical and numerical columns
        self.categorical_cols = ['proto', 'service', 'state']
        self.numerical_cols = [col for col in feature_cols if col not in self.categorical_cols]
        
        # Handle categorical features
        if label_encoders is None:
            self.label_encoders = {}
            for col in self.categorical_cols:
                le = LabelEncoder()
                self.features_df[col] = le.fit_transform(self.features_df[col].astype(str))
                self.label_encoders[col] = le
        else:
            self.label_encoders = label_encoders
            for col in self.categorical_cols:
                # Handle unknown categories
                self.features_df[col] = self.features_df[col].astype(str)
                known_classes = set(self.label_encoders[col].classes_)
                self.features_df[col] = self.features_df[col].apply(
                    lambda x: x if x in known_classes else self.label_encoders[col].classes_[0]
                )
                self.features_df[col] = self.label_encoders[col].transform(self.features_df[col])
        
        # Handle numerical features
        if fit_transform:
            self.scaler = StandardScaler()
            self.features_df[self.numerical_cols] = self.scaler.fit_transform(
                self.features_df[self.numerical_cols]
            )
        elif scaler is not None:
            self.scaler = scaler
            self.features_df[self.numerical_cols] = self.scaler.transform(
                self.features_df[self.numerical_cols]
            )
        else:
            self.scaler = None
        
        # Convert to numpy array
        self.features = self.features_df.values.astype(np.float32)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]])[0]


def get_unsw_nb15_dataloaders(data_dir, batch_size=128, num_workers=4):
    """
    Simple function to get UNSW_NB15 dataloaders
    
    Args:
        data_dir: Path to the data directory (e.g., 'nids-structured-label-poisoning/data/nusw_nb15/')
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, test_loader, train_dataset (contains scaler and encoders)
    """
    import os
    
    # Construct file paths
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join('data_real', 'nusw', 'test.csv')
    
    # Create datasets
    train_dataset = UNSW_NB15_Dataset(train_path, fit_transform=True)
    test_dataset = UNSW_NB15_Dataset(
        test_path, 
        scaler=train_dataset.scaler,
        label_encoders=train_dataset.label_encoders,
        fit_transform=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset


class CIC_IDS2017_Dataset(Dataset):
    """Simple dataset class for CIC-IDS2017"""
    
    def __init__(self, csv_path, scaler=None, fit_transform=False):
        """
        Args:
            csv_path: Path to the CSV file
            scaler: StandardScaler for numerical features (pass fitted scaler for test set)
            fit_transform: If True, fit the scaler (for training set)
        """
        # Load data
        self.df = pd.read_csv(csv_path)
        
        # Separate features and label
        # Convert to binary: 0 = benign, 1 = attack (any non-zero label)
        self.labels = (self.df['label'].values != 0).astype(int)
        
        # Drop label column
        feature_cols = [col for col in self.df.columns if col != 'label']
        self.features_df = self.df[feature_cols].copy()
        
        # Handle numerical features (all features are numerical in CIC-IDS2017)
        if fit_transform:
            self.scaler = StandardScaler()
            self.features_df = pd.DataFrame(
                self.scaler.fit_transform(self.features_df),
                columns=self.features_df.columns
            )
        elif scaler is not None:
            self.scaler = scaler
            self.features_df = pd.DataFrame(
                self.scaler.transform(self.features_df),
                columns=self.features_df.columns
            )
        else:
            self.scaler = None
        
        # Convert to numpy array
        self.features = self.features_df.values.astype(np.float32)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]])[0]


def get_cic_ids2017_dataloaders(data_dir, batch_size=128, num_workers=4):
    """
    Simple function to get CIC-IDS2017 dataloaders
    
    Args:
        data_dir: Path to the data directory (e.g., 'nids-structured-label-poisoning/data/cic-ids-2017/processed/')
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, test_loader, train_dataset (contains scaler)
    """
    import os
    
    # Construct file paths
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join('data_real', 'cic', 'test.csv')
    
    # Create datasets
    train_dataset = CIC_IDS2017_Dataset(train_path, fit_transform=True)
    test_dataset = CIC_IDS2017_Dataset(
        test_path, 
        scaler=train_dataset.scaler,
        fit_transform=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset


class CUPID_Dataset(Dataset):
    """Simple dataset class for CUPID"""
    
    def __init__(self, csv_path, scaler=None, fit_transform=False):
        """
        Args:
            csv_path: Path to the CSV file
            scaler: StandardScaler for numerical features (pass fitted scaler for test set)
            fit_transform: If True, fit the scaler (for training set)
        """
        # Load data
        self.df = pd.read_csv(csv_path)
        
        # Separate features and label
        self.labels = self.df['label'].values
        
        # Drop metadata columns that shouldn't be used for training
        cols_to_drop = ['Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp', 'label']
        feature_cols = [col for col in self.df.columns if col not in cols_to_drop]
        self.features_df = self.df[feature_cols].copy()
        
        # Handle numerical features (all remaining features are numerical)
        if fit_transform:
            self.scaler = StandardScaler()
            self.features_df = pd.DataFrame(
                self.scaler.fit_transform(self.features_df),
                columns=self.features_df.columns
            )
        elif scaler is not None:
            self.scaler = scaler
            self.features_df = pd.DataFrame(
                self.scaler.transform(self.features_df),
                columns=self.features_df.columns
            )
        else:
            self.scaler = None
        
        # Convert to numpy array
        self.features = self.features_df.values.astype(np.float32)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]])[0]


def get_cupid_dataloaders(data_dir, batch_size=128, num_workers=4):
    """
    Simple function to get CUPID dataloaders
    
    Args:
        data_dir: Path to the data directory (e.g., 'nids-structured-label-poisoning/data/cupid/')
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, test_loader, train_dataset (contains scaler)
    """
    import os
    
    test_path = os.path.join('data_real', 'cupid', 'test.csv')
    train_path = os.path.join(data_dir, 'train.csv')
    
    # Create datasets
    train_dataset = CUPID_Dataset(train_path, fit_transform=True)
    test_dataset = CUPID_Dataset(
        test_path, 
        scaler=train_dataset.scaler,
        fit_transform=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset


class CIDDS_Dataset(Dataset):
    """Simple dataset class for CIDDS"""
    
    def __init__(self, csv_path, scaler=None, proto_encoder=None, fit_transform=False):
        """
        Args:
            csv_path: Path to the CSV file
            scaler: StandardScaler for numerical features (pass fitted scaler for test set)
            proto_encoder: LabelEncoder for the proto column
            fit_transform: If True, fit the scalers (for training set)
        """
        # Load data
        self.df = pd.read_csv(csv_path)
        
        # Create binary labels from attack_type: 'benign' -> 0, else -> 1
        self.labels = (self.df['label']).astype(int).values
        
        # Drop label and metadata columns
        cols_to_drop = ['original_label', 'label', 'attack_id']
        feature_cols = [col for col in self.df.columns if col not in cols_to_drop]
        self.features_df = self.df[feature_cols].copy()
        
        # Identify categorical and numerical columns
        self.categorical_col = 'proto'
        self.numerical_cols = [col for col in feature_cols if col != self.categorical_col]
        
        # Handle categorical feature (proto)
        if proto_encoder is None:
            self.proto_encoder = LabelEncoder()
            self.features_df[self.categorical_col] = self.proto_encoder.fit_transform(
                self.features_df[self.categorical_col].astype(str).str.strip()
            )
        else:
            self.proto_encoder = proto_encoder
            # Handle unknown categories
            self.features_df[self.categorical_col] = self.features_df[self.categorical_col].astype(str).str.strip()
            known_classes = set(self.proto_encoder.classes_)
            self.features_df[self.categorical_col] = self.features_df[self.categorical_col].apply(
                lambda x: x if x in known_classes else self.proto_encoder.classes_[0]
            )
            self.features_df[self.categorical_col] = self.proto_encoder.transform(self.features_df[self.categorical_col])
        
        # Handle numerical features
        if fit_transform:
            self.scaler = StandardScaler()
            self.features_df[self.numerical_cols] = self.scaler.fit_transform(
                self.features_df[self.numerical_cols]
            )
        elif scaler is not None:
            self.scaler = scaler
            self.features_df[self.numerical_cols] = self.scaler.transform(
                self.features_df[self.numerical_cols]
            )
        else:
            self.scaler = None
        
        # Convert to numpy array
        self.features = self.features_df.values.astype(np.float32)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]])[0]


def get_cidds_dataloaders(data_dir, batch_size=128, num_workers=4):
    """
    Simple function to get CIDDS dataloaders
    
    Args:
        data_dir: Path to the data directory (e.g., 'nids-structured-label-poisoning/data/cidds/')
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, test_loader, train_dataset (contains scaler and encoders)
    """
    import os
    
    # Construct file paths
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join('data_real', 'cidds', 'test.csv')
    
    # Create datasets
    train_dataset = CIDDS_Dataset(train_path, fit_transform=True)
    test_dataset = CIDDS_Dataset(
        test_path, 
        scaler=train_dataset.scaler,
        proto_encoder=train_dataset.proto_encoder,
        fit_transform=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset

