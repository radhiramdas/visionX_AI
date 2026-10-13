import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

CLASSES = [
    "AMD",
    "Cataract",
    "DR",
    "Glaucoma",
    "Hypertensive_Retinopathy",
    "Normal_Fundus",
    "Pathological_Myopia"
]

class LetterboxResize:
    """
    Resizes an image to a target size while maintaining aspect ratio,
    padding the remaining space with black.
    """
    def __init__(self, target_size):
        self.target_size = target_size if isinstance(target_size, tuple) else (target_size, target_size)

    def __call__(self, img):
        w, h = img.size
        target_w, target_h = self.target_size
        
        # Calculate scaling factor to fit within target dimensions
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize using Lanczos interpolation
        resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create a new black image canvas
        new_img = Image.new("RGB", self.target_size, (0, 0, 0))
        # Paste the resized image centered
        new_img.paste(resized_img, ((target_w - new_w) // 2, (target_h - new_h) // 2))
        return new_img


def get_transforms(image_size=224, is_train=True):
    """
    Builds the PyTorch transformation pipeline.
    """
    target_size = (image_size, image_size)
    
    if is_train:
        return transforms.Compose([
            LetterboxResize(target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            LetterboxResize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


class RetinalDataset(Dataset):
    """
    PyTorch Dataset for retinal fundus images organized in class-specific subfolders.
    """
    def __init__(self, root_dir, image_size=224, is_train=True, transform=None):
        self.root_dir = root_dir
        self.image_size = image_size
        self.is_train = is_train
        
        # Class mapping
        self.classes = CLASSES
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Gather all image paths and their labels
        self.image_records = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue
            
            # Find all jpeg images
            pattern = os.path.join(cls_dir, "*.jpeg")
            img_paths = glob.glob(pattern)
            for path in img_paths:
                self.image_records.append((path, self.class_to_idx[cls_name]))
                
        # Set transforms
        self.transform = transform if transform is not None else get_transforms(image_size, is_train)

    def __len__(self):
        return len(self.image_records)

    def __getitem__(self, idx):
        img_path, label = self.image_records[idx]
        
        # Load image as RGB
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback if image cannot be read
            print(f"Error reading image {img_path}: {e}")
            # Create a black dummy image
            image = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
            
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label
