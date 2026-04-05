#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
#

"""Transform functions for AID Scene Classification dataset."""

import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# ImageNet normalization
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

def get_train_transform():
    return transforms.Compose([
        transforms.Resize(160),
        transforms.RandomResizedCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        NORMALIZE
    ])

def get_test_transform():
    return transforms.Compose([
        transforms.Resize(160),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        NORMALIZE
    ])

def aid_transform_train(examples):
    """Transform batch for training."""
    transform = get_train_transform()
    
    # Handle both PIL Images and paths if stored in the dataset
    features = []
    for img in examples["image"]:
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img).convert("RGB")
        # If it's already a PIL image, just ensure it's RGB
        elif hasattr(img, 'convert'):
            img = img.convert("RGB")
            
        features.append(transform(img))
    
    examples["features"] = features
    # Ensure labels are long tensors
    examples["label"] = [torch.tensor(l, dtype=torch.long) for l in examples["label"]]
    return examples

def aid_transform_test(examples):
    """Transform batch for testing/validation."""
    transform = get_test_transform()
    
    features = []
    for img in examples["image"]:
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img).convert("RGB")
        elif hasattr(img, 'convert'):
            img = img.convert("RGB")
            
        features.append(transform(img))
    
    examples["features"] = features
    examples["label"] = [torch.tensor(l, dtype=torch.long) for l in examples["label"]]
    return examples

def get_aid_transforms():
    """Return the transform functions for the AID dataset."""
    return {
        "train": aid_transform_train,
        "test": aid_transform_test
    }
