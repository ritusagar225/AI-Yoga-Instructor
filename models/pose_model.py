import torch
import torch.nn as nn

class YogaPoseMLP(nn.Module):
    """
    Standard MLP classifier mapping 91 feature-engineered inputs
    to one of 23 yoga pose classes.
    """
    def __init__(self, input_dim=91, num_classes=23, hidden_dims=[128, 64], dropout_rate=0.2):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class YogaPoseLSTM(nn.Module):
    """
    Temporal pose classifier utilizing LSTM blocks to process a sliding window
    sequence of landmarks/features. Useful for Dynamic transitions.
    """
    def __init__(self, input_dim=91, hidden_dim=64, num_layers=2, num_classes=23, dropout_rate=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM input shape: (batch, sequence_len, input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0)) # out shape: (batch, seq_len, hidden_dim)
        
        # Pull output from the final timestamp of the sequence
        last_step_out = out[:, -1, :] # shape: (batch, hidden_dim)
        
        # Fully connected projection
        logits = self.fc(last_step_out)
        return logits
