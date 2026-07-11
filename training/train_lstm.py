import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pose_model import YogaPoseLSTM

class YogaSequenceDataset(Dataset):
    """
    Synthetic temporal sequence dataset.
    Generates sequences of shape (seq_len, feature_dim) representing smooth pose entries and holds.
    """
    def __init__(self, num_sequences=500, seq_len=30, feature_dim=91, num_classes=23):
        super().__init__()
        self.num_sequences = num_sequences
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        self.X = []
        self.y = []
        
        # Generate synthetic smooth temporal trajectory data
        for _ in range(num_sequences):
            label = np.random.randint(0, num_classes)
            # Create a base feature template for this class
            base_feature = np.random.normal(0, 1.0, size=(feature_dim,)).astype(np.float32)
            
            # Synthesize sequence with smooth drift/interpolation (representing entering/holding pose)
            seq = []
            for t in range(seq_len):
                # Interpolate from starting position to pose position
                alpha = t / (seq_len - 1)
                noise = np.random.normal(0, 0.05, size=(feature_dim,)).astype(np.float32)
                frame_feature = alpha * base_feature + (1 - alpha) * np.zeros_like(base_feature) + noise
                seq.append(frame_feature)
                
            self.X.append(np.array(seq))
            self.y.append(label)
            
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.int64)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


def train_lstm():
    print("="*60)
    print("TEMPORAL POSE CLASSIFIER TRAINING SUITE (LSTM)")
    print("="*60)
    
    seq_len = 30
    feature_dim = 91
    num_classes = 23
    
    # 1. Instantiate datasets and loaders
    train_dataset = YogaSequenceDataset(num_sequences=800, seq_len=seq_len, feature_dim=feature_dim)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YogaPoseLSTM(input_dim=feature_dim, hidden_dim=64, num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Beginning LSTM sequence training...")
    epochs = 15
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        corrects = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == targets.data)
            
        epoch_loss /= len(train_loader.dataset)
        acc = float(corrects) / len(train_loader.dataset)
        
        print(f"Epoch {epoch+1}/{epochs} - Sequence Loss: {epoch_loss:.4f} | Training Acc: {acc:.4f}")

    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, "pose_lstm.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"LSTM Temporal Weights saved successfully to: {weights_path}")

if __name__ == "__main__":
    train_lstm()
