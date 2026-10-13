import numpy as np
from PIL import Image
import torch
from torchvision import transforms

try:
    from model_loader import get_model
except ImportError:
    from backend.model_loader import get_model

# Exact 7 retinal condition classes matching dataset.py and training setup
CLASSES = [
    "AMD",
    "Cataract",
    "DR",
    "Glaucoma",
    "Hypertensive_Retinopathy",
    "Normal_Fundus",
    "Pathological_Myopia"
]

IMAGE_SIZE = 224


class LetterboxResize:
    """
    Resizes an image to a target size while maintaining aspect ratio,
    padding the remaining space with black.
    Matches the exact implementation used during training in dataset.py
    and verified in gradcam_test.py.
    """
    def __init__(self, target_size=224):
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


# Deterministic tensor transformation pipeline (NO training augmentations)
_tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def preprocess_image(image: Image.Image, device=None):
    """
    Preprocesses an input image for both inference and Grad-CAM alignment.
    
    Guarantees:
    - RGB conversion
    - LetterboxResize to 224x224
    - Returns both the input tensor for the model and the EXACT SAME
      letterboxed RGB array (normalized to [0.0, 1.0]) for Grad-CAM overlay.
      
    Args:
        image (PIL.Image.Image): Input image.
        device (torch.device, optional): Device to move the input tensor to.
        
    Returns:
        tuple: (input_tensor: torch.Tensor, rgb_image: np.ndarray)
    """
    # 1. Ensure RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 2. Letterbox resize maintaining aspect ratio with black padding
    letterbox = LetterboxResize(IMAGE_SIZE)
    letterboxed_image = letterbox(image)

    # 3. Image array for Grad-CAM visualization (float32 in [0.0, 1.0], shape (224, 224, 3))
    rgb_image = np.array(letterboxed_image).astype(np.float32) / 255.0

    # 4. Input tensor for model inference (shape (1, 3, 224, 224))
    input_tensor = _tensor_transform(letterboxed_image).unsqueeze(0)
    if device is not None:
        input_tensor = input_tensor.to(device)

    return input_tensor, rgb_image


def predict(image: Image.Image, model=None, device=None):
    """
    Performs inference on a PIL image.
    Preserves exact backward compatibility with Checkpoint 1.
    
    Args:
        image (PIL.Image.Image): Input retinal fundus image.
        model (torch.nn.Module, optional): Loaded model. If None, loaded via model_loader.
        device (torch.device, optional): Target device.
        
    Returns:
        tuple: (predicted_class: str, confidence: float)
    """
    predicted_class, confidence, _, _, _ = predict_with_details(image, model=model, device=device)
    return predicted_class, confidence


def predict_with_details(image: Image.Image, model=None, device=None):
    """
    Performs inference on a PIL image and returns all prediction details
    and preprocessed artifacts needed for Grad-CAM.
    
    Args:
        image (PIL.Image.Image): Input retinal fundus image.
        model (torch.nn.Module, optional): Loaded model. If None, loaded via model_loader.
        device (torch.device, optional): Target device.
        
    Returns:
        tuple: (
            predicted_class: str,
            confidence: float,
            predicted_idx: int,
            input_tensor: torch.Tensor,
            rgb_image: np.ndarray
        )
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a valid PIL Image instance.")

    if model is None or device is None:
        model, device = get_model()

    # Preprocess image into aligned tensor and visualization array
    input_tensor, rgb_image = preprocess_image(image, device=device)

    # Run model forward pass with gradients disabled for inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_idx].item() * 100.0

    predicted_class = CLASSES[predicted_idx]
    confidence_rounded = round(confidence, 2)

    return predicted_class, confidence_rounded, predicted_idx, input_tensor, rgb_image
