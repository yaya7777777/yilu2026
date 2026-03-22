import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
sys.stdout.flush()

print("Importing libraries...")
sys.stdout.flush()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm import create_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

print("Libraries imported successfully!")
sys.stdout.flush()

print(f"torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
sys.stdout.flush()

print("\nLoading model...")
sys.stdout.flush()

# Load pretrained ViT base model
try:
    model = create_model('vit_base_patch16_224', pretrained=True)
    print("Model loaded successfully!")
    sys.stdout.flush()
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    exit(1)

# Modify head: from 1000 classes to 10 classes
features = model.head.in_features
model.head = nn.Linear(features, 10)

# Linear Probing: freeze all parameters except Head
for param in model.parameters():
    param.requires_grad = False
for param in model.head.parameters():
    param.requires_grad = True
    
# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.head.parameters(), lr=1e-5)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

print("\nLoading CIFAR-10 dataset...")
sys.stdout.flush()

# Load CIFAR-10
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Dataset loaded successfully!")
sys.stdout.flush()

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return total_loss / total, correct / total

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

num_epochs = 1
lp_train_accs = []
lp_test_accs = []

print("\n=== Linear Probing ===")
sys.stdout.flush()

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    test_acc = evaluate(model, test_loader, device)
    lp_train_accs.append(train_acc)
    lp_test_accs.append(test_acc)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    sys.stdout.flush()

# Full Fine-tuning
for param in model.parameters():
    param.requires_grad = True
    
optimizer = optim.Adam(model.parameters(), lr=1e-7)

print("\n=== Full Fine-tuning ===")
sys.stdout.flush()

ft_train_accs = []
ft_test_accs = []

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    test_acc = evaluate(model, test_loader, device)
    ft_train_accs.append(train_acc)
    ft_test_accs.append(test_acc)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    sys.stdout.flush()
    
# Plot accuracy comparison
epochs = range(1, num_epochs+1)
plt.figure(figsize=(10, 5))
plt.plot(epochs, lp_test_accs, label='Linear Probing Test Acc')
plt.plot(epochs, ft_test_accs, label='Full Fine-tuning Test Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.legend()
plt.savefig('accuracy_comparison.png')
print("\nAccuracy comparison saved to accuracy_comparison.png")
sys.stdout.flush()

import cv2
from PIL import Image

# Global variable to store attention weights
attention_weights = None

# Hook function: extract attention weights after the last Transformer Block's Attention layer
def attention_hook(module, input, output):
    global attention_weights
    try:
        # timm 0.9.16 version Attention output format
        if isinstance(output, tuple) and len(output) == 2:
            attention_weights = output[1].detach().cpu()
        else:
            attention_weights = None
    except Exception as e:
        print(f"Hook error: {e}")
        attention_weights = None

# Find the last Transformer Block's Attention layer and register hook
last_attn_layer = model.blocks[-1].attn
last_attn_layer.register_forward_hook(attention_hook)

print("\n=== Generating Attention Heatmaps ===")
sys.stdout.flush()

# Get a few images from CIFAR-10 test set to generate attention heatmaps
num_samples = 3
sample_indices = [0, 1, 2]  # First 3 images

# Denormalize function for displaying images
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize tensor back to [0, 1] range"""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)

for idx in sample_indices:
    try:
        # Get image from test set
        image_tensor, label = test_dataset[idx]
        
        # Denormalize for display
        image_display = denormalize(image_tensor)
        image_display = (image_display.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Add batch dimension and move to device
        input_tensor = image_tensor.unsqueeze(0).to(device)
        
        # Forward pass
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Process weights: get CLS Token's attention to all Image Patches
        if attention_weights is not None:
            try:
                # Get CLS Token's attention to all Image Patches
                cls_attn = attention_weights[0, :, 0, 1:].mean(dim=0)
                # Reshape to 14x14
                attn_map = cls_attn.reshape(14, 14).numpy()
                # Upsample to 224x224
                attn_map = cv2.resize(attn_map, (224, 224))
                # Normalize
                attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
                # Generate heatmap
                heatmap = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                # Overlay
                superimposed = cv2.addWeighted(image_display, 0.6, heatmap, 0.4, 0)
                # Save
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(image_display)
                plt.title(f'Original Image (Label: {label})')
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.imshow(superimposed)
                plt.title('Attention Heatmap')
                plt.axis('off')
                plt.savefig(f'attention_heatmap_sample_{idx}.png')
                print(f"Attention heatmap saved to attention_heatmap_sample_{idx}.png")
                sys.stdout.flush()
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                sys.stdout.flush()
        else:
            print(f"Cannot get attention weights for sample {idx}")
            sys.stdout.flush()
    except Exception as e:
        print(f"Error processing sample {idx}: {e}")
        sys.stdout.flush()

print("\nProgram completed!")
sys.stdout.flush()
