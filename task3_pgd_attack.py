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
epsilon = 0.02
alpha = 0.005  # step size
num_steps = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========== Normalize settings ==========
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

normalize = transforms.Normalize(mean, std)
inv_normalize = transforms.Normalize(
    mean=[-m/s for m, s in zip(mean, std)],
    std=[1/s for s in std]
)

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

# ========== Load model ==========
model = torchvision.models.resnet34(weights='IMAGENET1K_V1').to(device)
model.eval()

# ========== Load dataset ==========
dataset_path = "./TestDataSet"
adv_output_dir = "./Adversarial_Test_Set_2"
os.makedirs(adv_output_dir, exist_ok=True)

dataset = ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# ========== Label mappings ==========
with open(os.path.join(dataset_path, 'labels_list.json'), 'r') as f:
    lines = json.load(f)

idx_to_class = {idx: label.split(": ")[1] for idx, label in enumerate(lines)}
imagenet_categories = ResNet34_Weights.IMAGENET1K_V1.meta["categories"]
classname_to_idx = {name: idx for idx, name in enumerate(imagenet_categories)}

# ========== ε and α scaled ==========
epsilon_scaled = torch.tensor([epsilon / s for s in std], device=device).view(1, 3, 1, 1)
alpha_scaled = torch.tensor([alpha / s for s in std], device=device).view(1, 3, 1, 1)

# ========== PGD Attack ==========
def pgd_attack(image, label_idx, model, epsilon_scaled, alpha_scaled, steps):
    ori = image.clone().detach()
    pert = image.clone().detach()

    for _ in range(steps):
        pert.requires_grad = True
        out = model(pert)
        loss = torch.nn.functional.cross_entropy(out, label_idx)
        model.zero_grad()
        loss.backward()
        grad = pert.grad.data
        pert = pert + alpha_scaled * grad.sign()
        # projection
        delta = torch.clamp(pert - ori, min=-epsilon_scaled, max=epsilon_scaled)
        pert = (ori + delta).detach()

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

    adv_img = pgd_attack(img, imagenet_label, model, epsilon_scaled, alpha_scaled, num_steps)

    # check L_inf
    raw_img = inv_normalize(img.squeeze()).clamp(0, 1)
    raw_adv = inv_normalize(adv_img.squeeze()).clamp(0, 1)
    l_inf = (raw_adv - raw_img).abs().max().item()
    assert l_inf <= epsilon + 1e-4

    # save
    class_dir = os.path.join(adv_output_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    save_path = os.path.join(class_dir, f"img_{idx:04d}.png")
    transforms.ToPILImage()(raw_adv.cpu()).save(save_path)

    # evaluate
    pred = model(adv_img)
    _, top5 = pred.topk(5, dim=1)
    top1_correct += (top5[:, 0] == imagenet_label).sum().item()
    top5_correct += int(imagenet_label in top5[0])
    total += 1

    # visualize
    if visualized < 5 and top5[:, 0] != imagenet_label:
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        axs[0].imshow(raw_img.permute(1, 2, 0).detach().cpu().numpy())
        axs[0].set_title("Original")
        axs[0].axis("off")

        axs[1].imshow(raw_adv.permute(1, 2, 0).detach().cpu().numpy())
        axs[1].set_title("PGD Adversarial")
        axs[1].axis("off")

        plt.tight_layout()
        plt.savefig(f"pgd_example_{idx}.png")
        plt.close()
        visualized += 1

# ========== Report ==========
print("\n✅ PGD Attack Completed")
print(f"🎯 Top-1 Accuracy: {top1_correct / total * 100:.2f}%")
print(f"🎯 Top-5 Accuracy: {top5_correct / total * 100:.2f}%")
