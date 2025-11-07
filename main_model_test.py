import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import numpy as np
import os

# ======================================================
# CONFIGURATION
# ======================================================
test_dir = r"ecg_images/test"  # <-- apna test folder path
model_path = r"D:\ECG Model\resnet18_ecg_finetuned_simple.pth"  # <-- fine-tuned model ka path
num_classes = 4
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Using device: {device}")

# ======================================================
# DATA PREPROCESSING (same as training)
# ======================================================
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# ======================================================
# LOAD TEST DATASET
# ======================================================
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class_names = test_dataset.classes
print(f"📂 Found {len(test_dataset)} test images across {len(class_names)} classes: {class_names}")

# ======================================================
# LOAD MODEL (same architecture as training)
# ======================================================
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load weights
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

print("✅ Fine-tuned model loaded successfully for testing.\n")

# ======================================================
# TESTING LOOP
# ======================================================
y_true = []
y_pred = []
y_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(probs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_probs.extend(probs.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_probs = np.array(y_probs)

# ======================================================
# EVALUATION METRICS
# ======================================================
acc = np.mean(y_true == y_pred) * 100
print(f"🎯 Test Accuracy: {acc:.2f}%\n")

print("📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

print("🧮 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# ======================================================
# ROC-AUC (Multi-class)
# ======================================================
try:
    # Convert true labels to one-hot encoding
    y_true_onehot = np.zeros((y_true.shape[0], num_classes))
    y_true_onehot[np.arange(y_true.shape[0]), y_true] = 1

    roc_auc = roc_auc_score(y_true_onehot, y_probs, multi_class='ovr')
    print(f"\n📈 ROC-AUC Score: {roc_auc:.4f}")
except Exception as e:
    print(f"⚠️ ROC-AUC skipped (error: {e})")

# ======================================================
# OVERFITTING INDICATION
# ======================================================
if acc > 95:
    print("\n✅ Model generalizes well (no major overfitting detected).")
elif 85 <= acc <= 95:
    print("\n⚠️ Slight overfitting possible — consider more augmentation or dropout.")
else:
    print("\n❌ Model struggling — try longer training or more balanced data.")
