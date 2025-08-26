# src/train.py (snippet)
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

from data_loader import make_rwf2000_loaders

def main():
    loaders = make_rwf2000_loaders(
        root_dir="data/video/rwf2000",
        batch_size=8,
        num_workers=2,
        clip_len=16,
        resize=112,
    )

    model = r3d_18(weights="KINETICS400_V1")  # or None if you don’t want pretrained
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)  # x: [B, C, T, H, W]
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
        train_acc = correct / total
        train_loss = loss_sum / total

        # ---- validation ----
        model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for x, y in loaders["val"]:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                loss_sum += loss.item() * x.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += x.size(0)
        val_acc = correct / total
        val_loss = loss_sum / total

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f}")

if __name__ == "__main__":
    main()
