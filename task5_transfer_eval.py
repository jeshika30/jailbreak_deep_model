import torch
import torchvision
import torchvision.transforms as transforms
import os
import json
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models import vgg16, VGG16_Weights

# ========== Config ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Four datasets to evaluate
data_roots = {
    "Original": "./TestDataSet",
    "FGSM": "./Adversarial_Test_Set_1",
    "PGD": "./Adversarial_Test_Set_2",
    "Patch": "./Adversarial_Test_Set_3_Strong"
}

# ImageNet normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# ========== Load model: VGG16 ==========
model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device)
model.eval()

imagenet_categories = VGG16_Weights.IMAGENET1K_V1.meta["categories"]
classname_to_idx = {name: idx for idx, name in enumerate(imagenet_categories)}

# ========== Load label mapping ==========
with open(os.path.join(data_roots["Original"], 'labels_list.json'), 'r') as f:
    lines = json.load(f)
idx_to_class = {idx: label.split(": ")[1] for idx, label in enumerate(lines)}

# ========== Evaluation Function ==========
def evaluate_dataset(name, path):
    print(f"\n🔍 Evaluating: {name}")
    dataset = ImageFolder(root=path, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    top1_correct = 0
    top5_correct = 0
    total = 0

    for img, label in tqdm(loader):
        img = img.to(device)
        label_name = idx_to_class[label.item()]
        imagenet_label = torch.tensor([classname_to_idx[label_name]]).to(device)

        with torch.no_grad():
            out = model(img)
            _, top5 = out.topk(5, dim=1)
            top1_correct += (top5[:, 0] == imagenet_label).sum().item()
            top5_correct += (top5[0] == imagenet_label).sum().item()
            total += 1

    top1 = top1_correct / total * 100
    top5 = top5_correct / total * 100
    print(f"🎯 Top-1 Accuracy: {top1:.2f}%")
    print(f"🎯 Top-5 Accuracy: {top5:.2f}%")
    return top1, top5

# ========== Main ==========
results = {}
for name, path in data_roots.items():
    top1, top5 = evaluate_dataset(name, path)
    results[name] = {"top1": top1, "top5": top5}

# ========== ASCII Table Summary ==========
print("\n📊 Final Results on VGG16:")
print("-" * 45)
print(f"{'Dataset':<15} | {'Top-1 (%)':>10} | {'Top-5 (%)':>10}")
print("-" * 45)
for name, scores in results.items():
    print(f"{name:<15} | {scores['top1']:>10.2f} | {scores['top5']:>10.2f}")
print("-" * 45)
