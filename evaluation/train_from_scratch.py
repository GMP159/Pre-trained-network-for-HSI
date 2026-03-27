#!/usr/bin/env python
"""
Baseline: Train from scratch (no pre-training)
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
PATCH_DIR = "data/apple_patches_64x64_labeled"
BATCH_SIZE = 32
NUM_EPOCHS = 100
LR = 3e-4
NUM_CLASSES = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*80)
print("BASELINE - Training from Scratch")
print("="*80)

# Load datasets
train_dataset = ApplePatchDataset(PATCH_DIR, split='train', test_size=0.2, random_state=42)
test_dataset = ApplePatchDataset(PATCH_DIR, split='test', test_size=0.2, random_state=42)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Initialize random model
print("\nInitializing random model...")
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
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total trainable parameters: {total_params:,}\n")

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# Training loop
best_acc = 0.0

for epoch in range(NUM_EPOCHS):
    model.train()
    train_correct = 0
    train_total = 0
    
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.permute(0, 2, 3, 1)
        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
        
        logits = model(batch_X, mode='classification')
        loss = criterion(logits, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, pred = torch.max(logits, 1)
        train_correct += (pred == batch_y).sum().item()
        train_total += batch_y.size(0)
    
    train_acc = train_correct / train_total
    
    # Test
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.permute(0, 2, 3, 1)
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            logits = model(batch_X, mode='classification')
            _, pred = torch.max(logits, 1)
            test_correct += (pred == batch_y).sum().item()
            test_total += batch_y.size(0)
    
    test_acc = test_correct / test_total
    scheduler.step()
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")
    
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'outputs/apple_scratch_best.pth')

print(f"\nBest Test Accuracy: {best_acc*100:.2f}%")

Path("outputs/baseline_scratch_results.txt").write_text(
    f"Baseline Results\nBest Test Accuracy: {best_acc*100:.2f}%\nTotal Parameters: {total_params:,}\n"
)
