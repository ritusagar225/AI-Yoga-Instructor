import os
import sys
import json
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Ensure root folder is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pose_model import YogaPoseMLP
from datasets.loader import YogaDataset
from core.classifier import POSE_CLASSES

def run_evaluation(weights_path="weights/pose_mlp.pth", test_data_path="data/test/test.parquet"):
    print("="*60)
    print("MODEL EVALUATION & METRICS REPORT")
    print("="*60)
    
    if not os.path.exists(weights_path):
        print(f"Error: Trained weights file not found at {weights_path}. Please train the model first.")
        return
        
    if not os.path.exists(test_data_path):
        print(f"Error: Test dataset not found at {test_data_path}. Please run train_mlp.py first.")
        return

    # 1. Load Dataset
    dataset = YogaDataset(test_data_path, augment=False)
    loader = DataLoader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)

    # 2. Setup Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YogaPoseMLP(input_dim=dataset.X.shape[1], num_classes=23).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # 3. Model Inference
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(targets.numpy())

    # 4. Compute Metrics
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=POSE_CLASSES, output_dict=True)
    report_str = classification_report(y_true, y_pred, target_names=POSE_CLASSES)
    
    cm = confusion_matrix(y_true, y_pred)

    print(report_str)
    print(f"\nOverall Accuracy: {acc:.4f}")
    
    # 5. Save Evaluation Report
    eval_dir = "evaluation"
    os.makedirs(eval_dir, exist_ok=True)
    metrics_path = os.path.join(eval_dir, "metrics_report.json")
    
    summary_metrics = {
        "accuracy": acc,
        "macro_avg": report["macro avg"],
        "weighted_avg": report["weighted avg"],
        "confusion_matrix": cm.tolist()
    }
    
    with open(metrics_path, "w") as f:
        json.dump(summary_metrics, f, indent=2)
        
    print(f"\nEvaluation summary saved to: {metrics_path}")

if __name__ == "__main__":
    run_evaluation()
