import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# ----------------------------
# Testing after training
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_dir = r"ECG_Image_data/test"
img_size = 224

# ✅ Image preprocessing
data_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.Grayscale(num_output_channels=1),
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ✅ Dataset & DataLoader
test_dataset = datasets.ImageFolder(root=test_dir, transform=data_transforms)
num_classes = len(test_dataset.classes)
print(f"📂 Total images: {len(test_dataset)}, Classes: {test_dataset.classes}")

test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# ✅ Model setup
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("resnet18_ecg_base.pth", map_location=device))
model = model.to(device)
model.eval()

# ✅ Testing with tqdm progress bar
test_correct, test_total = 0, 0

print("\n🚀 Testing in progress...\n")
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating", unit="batch"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

# ✅ Final Results
test_acc = 100 * test_correct / test_total
print(f"\n🧾 Final Test Accuracy: {test_acc:.2f}%")
