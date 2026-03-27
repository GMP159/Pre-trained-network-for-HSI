#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import argparse

# Add project root so "src2" is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../hsi_project
sys.path.insert(0, str(PROJECT_ROOT))

from src2.evaluation.coffee_dataset import CoffeePatchDataset
from src2.models.masked_sst import MaskedSST


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/pretrain_enhanced/mae/checkpoint_best.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    PATCH_DIR = "data/patches_64x64_geotiff_2"
    NUM_CLASSES = 3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("LINEAR PROBING on COFFEE - MAE Pre-trained Encoder")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Use pretrained: {args.use_pretrained}")
    print(f"Device: {DEVICE}\n")

    print("Loading datasets...")
    train_dataset = CoffeePatchDataset(PATCH_DIR, split="train", test_size=0.2)
    test_dataset = CoffeePatchDataset(PATCH_DIR, split="test", test_size=0.2)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    print("\nBuilding MaskedSST encoder...")
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
        num_classes=NUM_CLASSES,
    )

    if args.use_pretrained:
        print("Loading MAE pre-trained weights (encoder only)...")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state_dict = ckpt["model_state_dict"]
        encoder_state = {
            k: v for k, v in state_dict.items()
            if not k.startswith("classification_head")
        }
        model.load_state_dict(encoder_state, strict=False)
    else:
        print("Using randomly initialized encoder (no pretraining).")

    encoder_params = []
    for name, param in model.named_parameters():
        if not name.startswith("classification_head"):
            param.requires_grad = False
            encoder_params.append(param)
    model.eval()
    model = model.to(DEVICE)
    print(f"Encoder frozen. Parameters: {sum(p.numel() for p in encoder_params):,}")

    classifier = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES),
    ).to(DEVICE)
    trainable_params = sum(p.numel() for p in classifier.parameters())
    print(f"Classifier parameters: {trainable_params:,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    results_file = PROJECT_ROOT / "outputs/coffee_linear_probe_results.txt"
    results_file.parent.mkdir(exist_ok=True, parents=True)

    for epoch in range(args.epochs):
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.permute(0, 2, 3, 1)
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

            with torch.no_grad():
                features, _ = model.forward_encoder(batch_X, apply_masking=False)
                features = features.mean(dim=(1, 2))

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

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Acc: {train_acc*100:.2f}% | "
            f"Test Acc: {test_acc*100:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            out_path = (
                PROJECT_ROOT / "outputs/coffee_classifier_mae_best.pth"
                if args.use_pretrained
                else PROJECT_ROOT / "outputs/coffee_classifier_random_best.pth"
            )
            torch.save(classifier.state_dict(), out_path)
            print(f"  -> Best model saved ({out_path}, acc: {best_acc*100:.2f}%)")

    print("\n" + "=" * 80)
    print("Linear Probing Complete!")
    print(f"Best Test Accuracy: {best_acc*100:.2f}%")
    print("=" * 80)

    with open(results_file, "a") as f:
        f.write(f"Use_pretrained={args.use_pretrained}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Best Test Accuracy: {best_acc*100:.2f}%\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n")
        f.write("-" * 80 + "\n")


if __name__ == "__main__":
    main()
