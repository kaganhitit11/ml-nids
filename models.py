import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """
    Simple MLP for binary classification on NIDS datasets.
    Works with all 4 datasets: UNSW-NB15, CIC-IDS2017, CUPID, CIDDS
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_classes=2, dropout=0.3):
        """
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer size (default 128)
            num_classes: Number of output classes (default 2 for binary)
            dropout: Dropout rate (default 0.3)
        """
        super(SimpleMLP, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            logits: Output tensor of shape (batch_size, num_classes)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SimpleRNN(nn.Module):
    """
    Simple RNN for binary classification on NIDS datasets.
    Works with all 4 datasets: UNSW-NB15, CIC-IDS2017, CUPID, CIDDS
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_classes=2, dropout=0.3, seq_len=None):
        """
        Args:
            input_dim: Number of input features
            hidden_dim: RNN hidden size (default 64)
            num_classes: Number of output classes (default 2 for binary)
            dropout: Dropout rate (default 0.3)
            seq_len: Sequence length for reshaping features (auto-computed if None)
        """
        super(SimpleRNN, self).__init__()
        
        # Auto-compute sequence length to evenly divide input_dim
        if seq_len is None:
            # Try to find a reasonable sequence length
            for s in [10, 8, 7, 6, 5, 4, 3, 2, 1]:
                if input_dim % s == 0:
                    seq_len = s
                    break
        
        self.seq_len = seq_len
        self.feature_dim = input_dim // seq_len
        
        self.rnn = nn.RNN(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0 if dropout == 0 else dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            logits: Output tensor of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # Reshape flat features into sequence: (batch, input_dim) -> (batch, seq_len, feature_dim)
        x = x.view(batch_size, self.seq_len, self.feature_dim)
        
        # RNN forward pass
        # out: (batch, seq_len, hidden_dim)
        # h_n: (1, batch, hidden_dim)
        out, h_n = self.rnn(x)
        
        # Take the last hidden state
        x = h_n.squeeze(0)  # (batch, hidden_dim)
        
        # Dropout and classification
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class SimpleCNN(nn.Module):
    """
    Simple 1D CNN for binary classification on NIDS datasets.
    Works with all 4 datasets: UNSW-NB15, CIC-IDS2017, CUPID, CIDDS
    """
    
    def __init__(self, input_dim, num_classes=2, dropout=0.3):
        """
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes (default 2 for binary)
            dropout: Dropout rate (default 0.3)
        """
        super(SimpleCNN, self).__init__()
        
        # Reshape features to (batch, 1, input_dim) for 1D convolution
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        
        # Calculate flattened size after convolutions and pooling
        # After first pool: input_dim // 2
        # After second pool: input_dim // 4
        self.flatten_size = 32 * (input_dim // 4)
        
        self.fc = nn.Linear(self.flatten_size, num_classes)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            logits: Output tensor of shape (batch_size, num_classes)
        """
        # Reshape to (batch, channels=1, input_dim) for 1D convolution
        x = x.unsqueeze(1)  # (batch, 1, input_dim)
        
        # First conv block
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten and classify
        x = x.view(x.size(0), -1)  # (batch, flatten_size)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def get_logistic_regression():
    """
    Returns a Logistic Regression classifier with default parameters.
    
    Returns:
        LogisticRegression model
    """
    return LogisticRegression(max_iter=1000, random_state=42)


def get_random_forest():
    """
    Returns a Random Forest classifier with default parameters.
    
    Returns:
        RandomForestClassifier model
    """
    return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)


