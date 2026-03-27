#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import argparse

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src2.evaluation.apple_dataset import ApplePatchDataset
from src2.models.masked_sst import MaskedSST

CONTRASTIVE_CHECKPOINT = "outputs/pretrain_enhanced/contrastive/checkpoint_best.pth"
PATCH_DIR = "data/apple_patches_64x64_labeled"
BATCH_SIZE = 8
NUM_EPOCHS = 100
LR = 1e-4
NUM_CLASSES = 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_pretrained",
        action="store_true",
        help="Load contrastive checkpoint before fine-tuning",
    )
    args = parser.parse_args()

    train_dataset = ApplePatchDataset(
        PATCH_DIR, split="train", test_size=0.2, random_state=42
    )
    test_dataset = ApplePatchDataset(
        PATCH_DIR, split="test", test_size=0.2, random_state=42
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    print("Building model...")
    model = MaskedSST(
        img_size=64,
        patch_h=2,
        patch_w=2,
        patch_c=16,
        in_channels=256,
        embed_dim=128,
        depth=4,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1,
        num_classes=NUM_CLASSES,
    )

    if args.use_pretrained:
        print("Loading pre-trained contrastive encoder...")
        checkpoint = torch.load(CONTRASTIVE_CHECKPOINT, map_location="cpu")
        state_dict = checkpoint["model_state_dict"]
        state_dict_encoder = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith("classification_head")
        }
        model.load_state_dict(state_dict_encoder, strict=False)
        print("Loaded contrastive encoder.")
    else:
        print("Training full model from random initialization.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = True

    print("Fine-tuning FULL model")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS)

    best_acc = 0.0
    best_path = (
        "outputs/finetune_contrastive_best.pth"
        if args.use_pretrained
        else "outputs/finetune_contrastive_random_best.pth"
    )

    print("Starting fine-tuning...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_correct = 0
        train_total = 0

        for spectral, labels in train_loader:
            spectral, labels = spectral.to(device), labels.to(device)

            # (B, C, H, W) -> (B, H, W, C)
            spectral = spectral.permute(0, 2, 3, 1)

            optimizer.zero_grad()
            outputs = model(spectral)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * train_correct / train_total

        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for spectral, labels in test_loader:
                spectral, labels = spectral.to(device), labels.to(device)
                spectral = spectral.permute(0, 2, 3, 1)

                outputs = model(spectral)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = 100.0 * test_correct / test_total

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_path)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{NUM_EPOCHS} - "
                f"Train: {train_acc:.2f}% - Test: {test_acc:.2f}% - Best: {best_acc:.2f}%"
            )

        scheduler.step()

    print(f"Complete! Best: {best_acc:.2f}% ({'pretrained' if args.use_pretrained else 'random'})")


if __name__ == "__main__":
    main()
