import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models import ResNet34_Weights
from PIL import Image

# ====== Config ======
epsilon_raw = 0.02  # in raw [0,1] scale
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====== Load model ======
model = torchvision.models.resnet34(weights='IMAGENET1K_V1').to(device)
model.eval()

# ====== Transforms ======
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

normalize = transforms.Normalize(mean, std)
inv_normalize = transforms.Normalize(
    mean=[-m / s for m, s in zip(mean, std)],
    std=[1 / s for s in std]
)

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

# ====== Dataset ======
dataset_path = "./TestDataSet"
adv_output_dir = "./Adversarial_Test_Set_1"
os.makedirs(adv_output_dir, exist_ok=True)

dataset = ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# ====== Label mapping ======
with open(os.path.join(dataset_path, 'labels_list.json'), 'r') as f:
    label_lines = json.load(f)

idx_to_class = {idx: label.split(": ")[1] for idx, label in enumerate(label_lines)}
imagenet_categories = ResNet34_Weights.IMAGENET1K_V1.meta["categories"]
classname_to_idx = {name: idx for idx, name in enumerate(imagenet_categories)}

# ====== Epsilon scaling ======
epsilon_scaled = torch.tensor([epsilon_raw / s for s in std], device=device).view(1, 3, 1, 1)

# ====== FGSM Attack ======
def fgsm_attack(image, epsilon_scaled, grad):
    return (image + epsilon_scaled * grad.sign()).detach()

top1_correct = 0
top5_correct = 0
total = 0
visualized = 0

for idx, (img, label) in enumerate(tqdm(dataloader)):
    img = img.to(device).requires_grad_()
    label = label.to(device)
    class_name = dataset.classes[label.item()]
    label_name = idx_to_class[label.item()]
    imagenet_label = torch.tensor([classname_to_idx[label_name]]).to(device)

    out = model(img)
    loss = torch.nn.functional.cross_entropy(out, imagenet_label)
    model.zero_grad()
    loss.backward()

    adv_img = fgsm_attack(img, epsilon_scaled, img.grad.data)

    # check L∞
    raw_img = inv_normalize(img.squeeze()).clamp(0, 1)
    raw_adv = inv_normalize(adv_img.squeeze()).clamp(0, 1)
    l_inf = (raw_adv - raw_img).abs().max().item()
    assert l_inf <= epsilon_raw + 1e-4

    # save
    class_dir = os.path.join(adv_output_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    filename = os.path.join(class_dir, f"img_{idx:04d}.png")
    transforms.ToPILImage()(raw_adv.cpu()).save(filename)

    # eval
    pred = model(adv_img)
    _, top5 = pred.topk(5, dim=1)
    top1_correct += (top5[:, 0] == imagenet_label).sum().item()
    top5_correct += int(imagenet_label in top5[0])
    total += 1

    if visualized < 5 and top5[:, 0] != imagenet_label:
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        axs[0].imshow(raw_img.permute(1, 2, 0).detach().cpu().numpy())
        axs[0].set_title("Original")
        axs[0].axis("off")

        axs[1].imshow(raw_adv.permute(1, 2, 0).detach().cpu().numpy())
        axs[1].set_title("Adversarial")
        axs[1].axis("off")

        plt.tight_layout()
        plt.savefig(f"adv_example_{idx}.png")
        plt.close()
        visualized += 1

# ====== Final Accuracy ======
top1_acc = top1_correct / total * 100
top5_acc = top5_correct / total * 100
print(f"\n✅ FGSM Done. ε={epsilon_raw}")
print(f"🎯 Top-1 Accuracy: {top1_acc:.2f}%")
print(f"🎯 Top-5 Accuracy: {top5_acc:.2f}%")
