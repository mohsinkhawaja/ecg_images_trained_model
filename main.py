import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os, time

def main():
    # ----------------------------
    # Device setup (GPU if available)
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # Speed up for fixed-size input
    print(f"🚀 Training on: {device}")

    if device.type == "cuda":
        print(f"💽 GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"💡 GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")

    # ----------------------------
    # Dataset paths
    # ----------------------------
    data_dir = r"ECG_Image_data"
    img_size = 224
    batch_size = 32  # ✅ You can safely increase to 8–16 on 940MX (test with nvidia-smi)
    num_epochs = 5

    # ----------------------------
    # Data transforms
    # ----------------------------
    data_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    # ----------------------------
    # Dataset & Dataloaders
    # ----------------------------
    full_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=data_transforms)
    num_classes = len(full_dataset.classes)
    print(f"📂 Total images: {len(full_dataset)}, Classes: {full_dataset.classes}")

    # 80/20 split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # ----------------------------
    # Model setup (ResNet18)
    # ----------------------------
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # ----------------------------
    # Loss & Optimizer
    # ----------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # ----------------------------
    # Training Loop
    # ----------------------------
    print("\n🧠 Starting Training...\n")
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # ----------------------------
        # Validation
        # ----------------------------
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                loss = criterion(val_outputs, val_labels)
                val_loss += loss.item()

                _, val_preds = torch.max(val_outputs, 1)
                val_correct += (val_preds == val_labels).sum().item()
                val_total += val_labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        epoch_time = time.time() - start_time

        print(f"📅 Epoch [{epoch+1}/{num_epochs}] "
              f"| 🧮 Train Loss: {avg_train_loss:.4f} | ✅ Train Acc: {train_acc:.2f}% "
              f"| 🧪 Val Loss: {avg_val_loss:.4f} | 🎯 Val Acc: {val_acc:.2f}% "
              f"| ⏱️ Time: {epoch_time/60:.2f} min")

    print("\n✅ Training Complete!")

    # ----------------------------
    # Save Model
    # ----------------------------
    model_save_path = "resnet18_ecg_base.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"💾 Model saved to: {model_save_path}")

    print("\n📥 To load and continue training later:")
    print(f"""
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, {num_classes})
model.load_state_dict(torch.load("{model_save_path}"))
model = model.to(device)
""")

# Entry point
if __name__ == "__main__":
    main()
