import os
import argparse
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report

from dataset import RetinalDataset, CLASSES
from model import build_model

def compute_class_weights(dataset):
    """
    Computes class weights based on the inverse frequency of training samples.
    """
    labels = [record[1] for record in dataset.image_records]
    class_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(CLASSES)
    
    weights = []
    for idx in range(num_classes):
        count = class_counts.get(idx, 1)
        # Standard inverse frequency formula: total / (num_classes * class_count)
        weight = total_samples / (num_classes * count)
        weights.append(weight)
        
    # Return as float tensor
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100
    return epoch_loss, epoch_acc


def validate_model(model, dataloader, criterion, device):
    """
    Validates the model and computes validation metrics (Loss, Accuracy, Macro F1).
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    val_loss = running_loss / len(all_targets)
    val_preds = np.array(all_preds)
    val_targets = np.array(all_targets)
    
    # Compute metrics
    accuracy = (np.sum(val_preds == val_targets) / len(val_targets)) * 100
    macro_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
    
    return val_loss, accuracy, macro_f1, val_targets, val_preds


def train_model(data_dir, image_size=224, batch_size=16, epochs=3, lr=0.001, 
                model_name="efficientnet_b0", model_path="best_model_efficientnet_b0.pth", 
                checkpoint_path="training_checkpoint_efficientnet_b0.pth",
                resume=False, dry_run=False, patience=3):
    """
    Main model training workflow.
    """
    if resume and not os.path.isfile(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' not found. Cannot resume training.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training using device: {device}")
    print(f"Model name: {model_name}")
    
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    # 1. Datasets
    print("Loading datasets...")
    train_dataset = RetinalDataset(train_dir, image_size=image_size, is_train=True)
    val_dataset = RetinalDataset(val_dir, image_size=image_size, is_train=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # 2. Compute Class Weights from full training dataset before any shrinking
    class_weights = compute_class_weights(train_dataset).to(device)
    print(f"Class weights: {[round(float(w), 4) for w in class_weights.tolist()]}")
    
    # Support Dry-Run (truncate dataset)
    if dry_run:
        print("Dry run requested. Shrinking datasets...")
        train_dataset.image_records = train_dataset.image_records[:128]
        val_dataset.image_records = val_dataset.image_records[:64]
        if epochs == 3 and not resume:
            epochs = 1
        print(f"Dry-run training samples: {len(train_dataset)}")
        print(f"Dry-run validation samples: {len(val_dataset)}")
        
    # 3. Dataloaders (num_workers=0 is safer on Windows)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 4. Loss Function (using full-dataset class weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 5. Model and Optimizer
    model = build_model(model_name=model_name, num_classes=len(CLASSES), pretrained=True, freeze_backbone=True)
    model = model.to(device)
    
    # Only optimize parameters with requires_grad=True (the new classifier head)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    # 6. Training State and Resume Support
    start_epoch = 0
    best_macro_f1 = 0.0
    epochs_no_improve = 0

    if resume:
        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            start_epoch = checkpoint['epoch']
            best_macro_f1 = checkpoint.get('best_macro_f1', 0.0)
            epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
            print(f"Resuming training from epoch {start_epoch + 1}")
            print(f"Loaded checkpoint state: completed {start_epoch}/{epochs} epochs, best Val Macro F1: {best_macro_f1:.4f}, epochs without improvement: {epochs_no_improve}")
            if start_epoch >= epochs:
                print(f"Training already completed ({start_epoch}/{epochs} epochs).")
                print(f"Training complete. Best validation Macro F1: {best_macro_f1:.4f}")
                return
        else:
            print(f"Error: Checkpoint file '{checkpoint_path}' not found. Cannot resume training.")
            return
    else:
        print("Starting a completely fresh training run from Epoch 1...")
        if os.path.isfile(checkpoint_path):
            print(f"Note: Existing checkpoint '{checkpoint_path}' ignored (starting fresh from Epoch 1).")

    # 7. Training Loop
    print("Starting training...")
    for epoch in range(start_epoch, epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = validate_model(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Macro F1: {val_f1:.4f}")
              
        # Early Stopping / Saving Best Model based on F1-score
        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            epochs_no_improve = 0
            # Save best model weights
            torch.save(model.state_dict(), model_path)
            print(f"--> Saved best model weights with Val Macro F1: {val_f1:.4f} to {model_path}")
        else:
            epochs_no_improve += 1

        # Save complete training checkpoint after each completed epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_macro_f1': best_macro_f1,
            'epochs_no_improve': epochs_no_improve,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"--> Checkpoint saved to {checkpoint_path}")

        # Check early stopping condition
        if epochs_no_improve >= patience and not dry_run:
            print(f"Early stopping triggered. No validation F1 improvement for {patience} epochs.")
            break
                
    print(f"Training complete. Best validation Macro F1: {best_macro_f1:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VisionX AI Model Training Script")
    parser.add_argument("--data-dir", type=str, default="PreparedData", help="Path to prepared splits directory")
    parser.add_argument("--image-size", type=int, default=224, help="Target image size")
    parser.add_argument("--batch-size", type=int, default=16, help="DataLoader batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--model-name", type=str, default="efficientnet_b0", help="Model name (e.g. efficientnet_b0, tv_resnet18)")
    parser.add_argument("--model-path", type=str, default="best_model_efficientnet_b0.pth", help="Path to save best weights")
    parser.add_argument("--checkpoint-path", type=str, default="training_checkpoint_efficientnet_b0.pth", help="Path to save/load training checkpoint")
    parser.add_argument("--resume", action="store_true", help="Resume training from existing checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Perform a fast 1-epoch execution on a tiny subset of data")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience epochs")
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        model_name=args.model_name,
        model_path=args.model_path,
        checkpoint_path=args.checkpoint_path,
        resume=args.resume,
        dry_run=args.dry_run,
        patience=args.patience
    )
