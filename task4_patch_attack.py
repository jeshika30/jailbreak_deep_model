import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models import ResNet34_Weights
from PIL import Image

# ========== Config ==========
epsilon = 0.5
alpha = 0.1
num_steps = 30
patch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========== Normalization ==========
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

# ========== Load model ==========
model = torchvision.models.resnet34(weights='IMAGENET1K_V1').to(device)
model.eval()

# ========== Dataset ==========
dataset_path = "./TestDataSet"
adv_output_dir = "./Adversarial_Test_Set_3_Strong"
os.makedirs(adv_output_dir, exist_ok=True)

dataset = ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# ========== Label mapping ==========
with open(os.path.join(dataset_path, 'labels_list.json'), 'r') as f:
    lines = json.load(f)

idx_to_class = {idx: label.split(": ")[1] for idx, label in enumerate(lines)}
imagenet_categories = ResNet34_Weights.IMAGENET1K_V1.meta["categories"]
classname_to_idx = {name: idx for idx, name in enumerate(imagenet_categories)}

# ========== ε/α scaling ==========
epsilon_scaled = torch.tensor([epsilon / s for s in std], device=device).view(1, 3, 1, 1)
alpha_scaled = torch.tensor([alpha / s for s in std], device=device).view(1, 3, 1, 1)

# ========== PGD Patch Attack (Untargeted) ==========
def pgd_patch_attack(image, label_idx, model, epsilon_scaled, alpha_scaled, steps, patch_size):
    ori = image.clone().detach()
    pert = image.clone().detach()
    _, _, H, W = image.shape

    top = np.random.randint(0, H - patch_size)
    left = np.random.randint(0, W - patch_size)

    for _ in range(steps):
        pert.requires_grad = True
        output = model(pert)
        loss = torch.nn.functional.cross_entropy(output, label_idx)
        model.zero_grad()
        loss.backward()
        grad = pert.grad.data

        update = alpha_scaled * grad.sign()
        mask = torch.zeros_like(update)
        mask[:, :, top:top+patch_size, left:left+patch_size] = 1.0
        update = update * mask

        pert = pert + update
        delta = torch.clamp(pert - ori, min=-epsilon_scaled, max=epsilon_scaled)
        pert = (ori + delta) * mask + ori * (1 - mask)
        pert = pert.detach()

    return pert

# ========== Evaluation ==========
top1_correct = 0
top5_correct = 0
total = 0
visualized = 0

for idx, (img, label) in enumerate(tqdm(dataloader)):
    img = img.to(device)
    label = label.to(device)
    class_name = dataset.classes[label.item()]
    label_name = idx_to_class[label.item()]
    imagenet_label = torch.tensor([classname_to_idx[label_name]]).to(device)

    adv_img = pgd_patch_attack(img, imagenet_label, model, epsilon_scaled, alpha_scaled, num_steps, patch_size)

    # Save adversarial image
    raw_adv = inv_normalize(adv_img.squeeze()).clamp(0, 1)
    class_dir = os.path.join(adv_output_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    save_path = os.path.join(class_dir, f"img_{idx:04d}.png")
    transforms.ToPILImage()(raw_adv.cpu()).save(save_path)

    # Evaluate
    pred = model(adv_img)
    _, top5 = pred.topk(5, dim=1)
    top1_correct += (top5[:, 0] == imagenet_label).sum().item()
    top5_correct += int(imagenet_label in top5[0])
    total += 1

    # Visualization
    if visualized < 5 and top5[:, 0] != imagenet_label:
        raw_img = inv_normalize(img.squeeze()).clamp(0, 1)
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        axs[0].imshow(raw_img.permute(1, 2, 0).detach().cpu().numpy())
        axs[0].set_title("Original")
        axs[0].axis("off")
        axs[1].imshow(raw_adv.permute(1, 2, 0).detach().cpu().numpy())
        axs[1].set_title("Strong Patch Attack")
        axs[1].axis("off")
        plt.tight_layout()
        plt.savefig(f"patch_strong_example_{idx}.png")
        plt.close()
        visualized += 1

# ========== Results ==========
print("\n✅ Strong Patch PGD Attack Completed")
print(f"🎯 Top-1 Accuracy: {top1_correct / total * 100:.2f}%")
print(f"🎯 Top-5 Accuracy: {top5_correct / total * 100:.2f}%")
