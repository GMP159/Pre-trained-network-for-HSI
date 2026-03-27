#!/usr/bin/env python
"""
Linear probing with MAE pre-trained encoder
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from apple_dataset import ApplePatchDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.masked_sst import MaskedSST

# Config
MAE_CHECKPOINT = "outputs/pretrain_enhanced/mae/checkpoint_epoch160.pth"
PATCH_DIR = "data/apple_patches_64x64_labeled"
BATCH_SIZE = 32
NUM_EPOCHS = 50
LR = 1e-3
NUM_CLASSES = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*80)
print("LINEAR PROBING - MAE Pre-trained Encoder")
print("="*80)
print(f"MAE checkpoint: {MAE_CHECKPOINT}")
print(f"Device: {DEVICE}\n")

# Load datasets
print("Loading datasets...")
train_dataset = ApplePatchDataset(PATCH_DIR, split='train', test_size=0.2, random_state=42)
test_dataset = ApplePatchDataset(PATCH_DIR, split='test', test_size=0.2, random_state=42)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Load MAE pre-trained model
print("\nLoading MAE pre-trained model...")
model = MaskedSST(
    img_size=64,
    patch_h=4,
    patch_w=4,
    patch_c=16,
    in_channels=256,
    embed_dim=128,
    depth=4,
    num_heads=8,
    mlp_ratio=4,
    dropout=0.1,
    num_classes=NUM_CLASSES
)

checkpoint = torch.load(MAE_CHECKPOINT, map_location='cpu')

# Filter out classification head (incompatible size)
state_dict = checkpoint['model_state_dict']
encoder_state = {k: v for k, v in state_dict.items() 
                 if not k.startswith('classification_head')}

# Load only encoder weights
model.load_state_dict(encoder_state, strict=False)

# Extract encoder components (freeze everything except classification head)
encoder_params = []
for name, param in model.named_parameters():
    if not name.startswith('classification_head'):
        param.requires_grad = False
        encoder_params.append(param)

model.eval()  # Set encoder to eval mode
model = model.to(DEVICE)

print(f"Encoder frozen. Parameters: {sum(p.numel() for p in encoder_params):,}")

# Create new classification head
classifier = nn.Sequential(
    nn.Linear(128, 256),  # embed_dim=128 from MaskedSST
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, NUM_CLASSES)
).to(DEVICE)

trainable_params = sum(p.numel() for p in classifier.parameters())
print(f"Classifier parameters: {trainable_params:,}\n")

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(classifier.parameters(), lr=LR, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# Training loop
best_acc = 0.0
results_file = Path("outputs/linear_probe_mae_results.txt")
results_file.parent.mkdir(exist_ok=True, parents=True)

for epoch in range(NUM_EPOCHS):
    # Train
    classifier.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_X, batch_y in train_loader:
        # Transpose to (B, H, W, C) format expected by MaskedSST
        batch_X = batch_X.permute(0, 2, 3, 1)  # (B, 256, 64, 64) -> (B, 64, 64, 256)
        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
        
        # Extract features from frozen encoder
        with torch.no_grad():
            features, _ = model.forward_encoder(batch_X, apply_masking=False)
            # features: (B, n_spatial, n_spectral, embed_dim) = (B, 256, 16, 128)
            # Global average pooling
            features = features.mean(dim=(1, 2))  # (B, 128)
        
        # Classify
        logits = classifier(features)
        loss = criterion(logits, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, pred = torch.max(logits, 1)
        train_correct += (pred == batch_y).sum().item()
        train_total += batch_y.size(0)
    
    train_acc = train_correct / train_total
    avg_train_loss = train_loss / len(train_loader)
    
    # Test
    classifier.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.permute(0, 2, 3, 1)
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            features, _ = model.forward_encoder(batch_X, apply_masking=False)
            features = features.mean(dim=(1, 2))
            logits = classifier(features)
            
            _, pred = torch.max(logits, 1)
            test_correct += (pred == batch_y).sum().item()
            test_total += batch_y.size(0)
    
    test_acc = test_correct / test_total
    scheduler.step()
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Train Acc: {train_acc*100:.2f}% | "
          f"Test Acc: {test_acc*100:.2f}%")
    
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(classifier.state_dict(), 'outputs/apple_classifier_mae_best.pth')
        print(f"  -> Best model saved (test acc: {best_acc*100:.2f}%)")

print(f"\n{'='*80}")
print(f"Linear Probing Complete!")
print(f"Best Test Accuracy: {best_acc*100:.2f}%")
print(f"{'='*80}")

with open(results_file, 'w') as f:
    f.write(f"MAE Linear Probing Results\n")
    f.write(f"{'='*80}\n")
    f.write(f"Checkpoint: {MAE_CHECKPOINT}\n")
    f.write(f"Best Test Accuracy: {best_acc*100:.2f}%\n")
    f.write(f"Trainable Parameters: {trainable_params:,}\n")
