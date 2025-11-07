import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
from collections import Counter

# ------------------------------------------------
# ✅ Configuration
# ------------------------------------------------
data_dir = r"ecg_images/train"
base_model_path = "resnet18_ecg_base.pth"
save_model_path = "resnet18_ecg_finetuned_simple.pth"

batch_size = 8
learning_rate = 1e-4
num_epochs = 10
img_size = 224
val_split = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ------------------------------------------------
# ✅ Data transforms
# ------------------------------------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
    transforms.Grayscale(num_output_channels=1),
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.Grayscale(num_output_channels=1),
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# ------------------------------------------------
# ✅ Dataset loading & splitting
# ------------------------------------------------
full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
num_classes = len(full_dataset.classes)
print(f"📂 Found {len(full_dataset)} images across {num_classes} classes: {full_dataset.classes}")

# Split train/val
val_size = int(len(full_dataset) * val_split)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"📊 Split: {train_size} train | {val_size} validation")

# ------------------------------------------------
# ✅ Model setup (load convolutional layers only)
# ------------------------------------------------
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load convolutional weights from base model
pretrained_dict = torch.load(base_model_path, map_location=device)
model_dict = model.state_dict()

pretrained_dict = {k: v for k, v in pretrained_dict.items()
                   if k in model_dict and v.size() == model_dict[k].size()}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
print("✅ Loaded convolutional layers from base model (FC skipped)")

# Freeze lower layers (to prevent overfitting)
for name, param in model.named_parameters():
    if "layer4" not in name and "fc" not in name:
        param.requires_grad = False

model = model.to(device)

# ------------------------------------------------
# ✅ Loss, Optimizer, Scheduler
# ------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

# ------------------------------------------------
# ✅ Training loop with validation
# ------------------------------------------------
best_val_acc = 0.0
print("\n🚀 Starting Training...\n")

for epoch in range(num_epochs):
    # ---------- TRAIN ----------
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    train_bar = tqdm(train_loader, desc=f"Train [{epoch+1}/{num_epochs}]", unit="batch")

    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        train_bar.set_postfix(loss=loss.item(), acc=100 * correct / total)

    train_acc = 100 * correct / total
    avg_train_loss = train_loss / len(train_loader)

    # ---------- VALIDATION ----------
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc=f"Val [{epoch+1}/{num_epochs}]", unit="batch")
        for images, labels in val_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            val_bar.set_postfix(loss=loss.item(), acc=100 * val_correct / val_total)

    val_acc = 100 * val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)

    print(f"📅 Epoch [{epoch+1}/{num_epochs}] | 🧮 Train Loss: {avg_train_loss:.4f} | ✅ Train Acc: {train_acc:.2f}% | 🧪 Val Loss: {avg_val_loss:.4f} | 🎯 Val Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_model_path)
        print("💾 Model improved & saved!")

    scheduler.step()

print(f"\n✅ Training completed! Best Val Accuracy: {best_val_acc:.2f}%")
