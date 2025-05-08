import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import json
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models import ResNet34_Weights

# ========== 1. Device ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========== 2. Load pretrained model ==========
model = torchvision.models.resnet34(weights='IMAGENET1K_V1')
model.eval()
model.to(device)

# ========== 3. Transforms ==========
mean_norms = np.array([0.485, 0.456, 0.406])
std_norms = np.array([0.229, 0.224, 0.225])
plain_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_norms, std=std_norms)
])

# ========== 4. Load dataset ==========
dataset_path = "./TestDataSet"  # <-- Replace with your dataset path
dataset = ImageFolder(root=dataset_path, transform=plain_transforms)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

# ========== 5. Load labels_list.json ==========
with open(os.path.join(dataset_path, 'labels_list.json'), 'r') as f:
    class_list = json.load(f)  # e.g., ["675: tram", ...]

# idx_to_class: dataset index → label name (e.g. 0 → "tram")
idx_to_class = {idx: class_id.split(": ")[1] for idx, class_id in enumerate(class_list)}

# ========== 6. Build WordNet class name to ImageNet index map ==========
imagenet_class_ids = ResNet34_Weights.IMAGENET1K_V1.meta["categories"]
classid_to_idx = {cls_id: idx for idx, cls_id in enumerate(imagenet_class_ids)}

# ========== 7. Evaluation ==========
top1_correct = 0
top5_correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, top5_preds = outputs.topk(5, dim=1)
        top1_preds = top5_preds[:, 0]

        # Map dataset labels → ImageNet class index
        correct_labels = torch.tensor([
            classid_to_idx[idx_to_class[labels[i].item()]]
            for i in range(len(labels))
        ], device=labels.device)

        # Top-1 and Top-5 match check
        top1_batch = (top1_preds == correct_labels)
        top5_batch = torch.tensor([
            correct_labels[i].item() in top5_preds[i]
            for i in range(len(labels))
        ], dtype=torch.bool, device=labels.device)

        top1_correct += top1_batch.sum().item()
        top5_correct += top5_batch.sum().item()
        total += labels.size(0)

# ========== 8. Report ==========
top1_acc = top1_correct / total
top5_acc = top5_correct / total

print(f"\nEvaluated {total} samples.")
print(f" Top-1 Accuracy: {top1_acc:.4f}")
print(f" Top-5 Accuracy: {top5_acc:.4f}")
