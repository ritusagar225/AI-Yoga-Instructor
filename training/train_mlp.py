import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
# Ensure root folder is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pose_model import YogaPoseMLP
from datasets.loader import YogaDataset, SyntheticDatasetGenerator
from core.classifier import POSE_CLASSES

def train_classifier():
    # 1. Build dataset splits if not already present
    data_dir = "data"
    train_path = os.path.join(data_dir, "train", "train.parquet")
    val_path = os.path.join(data_dir, "val", "val.parquet")
    test_path = os.path.join(data_dir, "test", "test.parquet")
    
    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        print("Dataset not found. Generating synthetic pose dataset...")
        generator = SyntheticDatasetGenerator()
        generator.build_dataset(samples_per_class=350, output_dir=data_dir)

    # 2. Instantiate datasets and loaders
    train_dataset = YogaDataset(train_path, augment=True)
    val_dataset = YogaDataset(val_path, augment=False)
    test_dataset = YogaDataset(test_path, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    print(f"Dataset loaded successfully. Input features shape: {train_dataset.X.shape[1]}")

    # 3. Model Setup
    input_dim = train_dataset.X.shape[1]
    num_classes = 23 # 23 yoga poses
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YogaPoseMLP(input_dim=input_dim, num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 4. Training Loop
    epochs = 40
    best_val_loss = float("inf")
    patience = 8
    patience_counter = 0
    
    print("Beginning model training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation evaluation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == targets.data)
                
        val_loss /= len(val_loader.dataset)
        val_acc = float(val_corrects) / len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Early Stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save temporary best checkpoint
            os.makedirs("weights", exist_ok=True)
            torch.save(model.state_dict(), "weights/pose_mlp_best.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Load best model weights
    model.load_state_dict(torch.load("weights/pose_mlp_best.pth"))
    model.eval()

    # 5. Temperature Scaling calibration on Validation Set
    print("Optimizing classification confidence calibration (Temperature Scaling)...")
    val_logits = []
    val_targets = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            val_logits.append(logits.cpu())
            val_targets.append(targets)
            
    val_logits = torch.cat(val_logits)
    val_targets = torch.cat(val_targets)
    
    # Grid search for optimal temperature scaling parameter
    best_temp = 1.0
    best_nll = float("inf")
    
    for t_val in np.arange(0.5, 3.0, 0.05):
        scaled_logits = val_logits / t_val
        nll = nn.CrossEntropyLoss()(scaled_logits, val_targets).item()
        if nll < best_nll:
            best_nll = nll
            best_temp = t_val
            
    print(f"Optimal Calibration Temperature: T = {best_temp:.2f} (Reduced NLL to {best_nll:.4f})")

    # 6. Evaluation on Test Set
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            # Apply scaled temperature calibration during forward pass
            logits = model(inputs) / best_temp
            _, preds = torch.max(logits, 1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(targets.numpy())
            
    test_acc = accuracy_score(test_targets, test_preds)
    print("\n" + "="*50)
    print(f"TEST SET PERFORMANCE REPORT (Accuracy: {test_acc:.4f})")
    print("="*50)
    print(classification_report(test_targets, test_preds, target_names=POSE_CLASSES))
    print("="*50)

    # 7. Save Final Checkpoint
    final_path = "weights/pose_mlp.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Calibrated model checkpoint saved to: {final_path}")
    
    # Clean up temp file
    if os.path.exists("weights/pose_mlp_best.pth"):
        os.remove("weights/pose_mlp_best.pth")

if __name__ == "__main__":
    train_classifier()
