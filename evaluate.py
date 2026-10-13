import os
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from dataset import RetinalDataset, CLASSES
from model import build_model

def evaluate_model(data_dir, image_size=128, batch_size=32, model_name="tv_resnet18", 
                   model_path="best_model.pth", output_dir="evaluation_results"):
    """
    Evaluates a trained model on the test split and saves metrics and plots.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    test_dir = os.path.join(data_dir, "test")
    
    # 1. Dataset & Dataloader
    print("Loading test dataset...")
    test_dataset = RetinalDataset(test_dir, image_size=image_size, is_train=False)
    print(f"Test samples: {len(test_dataset)}")
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 2. Build Model and Load Weights
    model = build_model(model_name=model_name, num_classes=len(CLASSES), pretrained=False, freeze_backbone=False)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint path not found: {model_path}")
        
    print(f"Loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 3. Inference Loop
    all_preds = []
    all_targets = []
    
    print("Running inference on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.numpy())
            
    val_preds = np.array(all_preds)
    val_targets = np.array(all_targets)
    
    # 4. Generate Classification Metrics
    print("Generating classification report...")
    report_str = classification_report(val_targets, val_preds, target_names=CLASSES, zero_division=0)
    report_dict = classification_report(val_targets, val_preds, target_names=CLASSES, output_dict=True, zero_division=0)
    
    print("\n--- Test Set Classification Report ---")
    print(report_str)
    
    # Add overall accuracy and macro averages directly to report_dict for convenience
    accuracy = (np.sum(val_preds == val_targets) / len(val_targets)) * 100
    print(f"Overall Test Accuracy: {accuracy:.2f}%")
    
    # Save text report
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report_str)
        f.write(f"\nOverall Test Accuracy: {accuracy:.2f}%\n")
        
    # Save json report
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(report_dict, f, indent=4)
    print(f"Saved classification metrics to {os.path.join(output_dir, 'metrics.json')}")
    
    # 5. Confusion Matrix Plotting
    print("Generating confusion matrix...")
    cm = confusion_matrix(val_targets, val_preds)
    
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=CLASSES, 
        yticklabels=CLASSES,
        cbar=True,
        square=True
    )
    plt.title("Confusion Matrix - VisionX AI (Test Split)", fontsize=14, pad=15)
    plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
    plt.ylabel("Actual Class", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save confusion matrix plot
    cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_plot_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VisionX AI Model Evaluation Script")
    parser.add_argument("--data-dir", type=str, default="PreparedData", help="Path to prepared splits directory")
    parser.add_argument("--image-size", type=int, default=128, help="Target image size")
    parser.add_argument("--batch-size", type=int, default=32, help="DataLoader batch size")
    parser.add_argument("--model-name", type=str, default="tv_resnet18", help="Model name (e.g. tv_resnet18, tv_mobilenetv3_small)")
    parser.add_argument("--model-path", type=str, default="best_model.pth", help="Path to trained weights")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Directory to save evaluation artifacts")
    
    args = parser.parse_args()
    
    evaluate_model(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        model_name=args.model_name,
        model_path=args.model_path,
        output_dir=args.output_dir
    )
