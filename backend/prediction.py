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


def validate_fundus_image(image: Image.Image) -> tuple:
    """
    Validates whether an uploaded image possesses the optical, color, and
    morphological characteristics of an authentic retinal fundus photograph.

    Rejects unrelated non-retinal inputs:
    - Diagrams, flowcharts, wireframes, sequence diagrams
    - Plain text documents, scanned pages, whiteboard captures
    - Blank, solid color, or heavily over/under-exposed images
    - Low-resolution thumbnails or extreme aspect ratios
    - Non-biological / non-retinal color profiles (e.g., severe blue dominance)

    Returns:
        tuple: (is_valid: bool, rejection_message: str)
    """
    if not isinstance(image, Image.Image):
        return False, "The uploaded file is not a valid image format."

    rgb = image.convert("RGB")
    arr = np.array(rgb, dtype=np.float32)
    h, w, _ = arr.shape

    # 1. Minimum resolution check (fundus cameras produce high-res scans)
    if w < 100 or h < 100:
        return False, "Image resolution is too low for clinical evaluation. Please upload a high-resolution retinal fundus scan."

    # 2. Aspect ratio check (fundus cameras produce roughly square or standard 4:3 / 3:2 fields)
    aspect = w / h
    if aspect < 0.45 or aspect > 2.2:
        return False, "Image aspect ratio does not match standard fundus photography. Please upload a standard retinal scan."

    mean_rgb = arr.mean(axis=(0, 1))
    overall_mean = float(np.mean(mean_rgb))

    # 3. Brightness check (Documents, whiteboards, flowcharts on white background)
    if overall_mean > 210:
        return False, "The uploaded image appears to be a document or diagram with a white background, not a retinal fundus scan. Please upload a genuine retinal fundus photograph."

    # 4. Extreme darkness / blank black images
    if overall_mean < 15:
        return False, "The uploaded image is too dark or underexposed. Retinal anatomical structures cannot be identified."

    # 5. Biological color saturation check (HSV)
    # Human retinal fundus photography contains rich hemoglobin and melanin pigments (sat > 20).
    # Monochrome diagrams, scanned documents, and wireframes have near-zero saturation.
    hsv = np.array(image.convert("HSV"), dtype=np.float32)
    sat_mean = float(hsv[:, :, 1].mean())
    if sat_mean < 8.0:
        return False, "The uploaded image lacks biological color saturation (monochrome/diagram/sketch detected). Please upload a true-color retinal fundus photograph."

    # 6. Bright corner check
    # Retinal fundus cameras produce circular optical fields with dark or padded periphery.
    # Rectangular documents and screenshots have bright white margins/corners.
    cw, ch = max(1, int(w * 0.05)), max(1, int(h * 0.05))
    corners = np.concatenate([
        arr[:ch, :cw].reshape(-1, 3),
        arr[:ch, -cw:].reshape(-1, 3),
        arr[-ch:, :cw].reshape(-1, 3),
        arr[-ch:, -cw:].reshape(-1, 3)
    ], axis=0)
    corner_mean = float(corners.mean())
    if corner_mean > 195:
        return False, "The uploaded image has bright white borders or corners typical of a diagram or screenshot. Fundus scans require a circular optical field."

    # 7. Anatomical contrast / texture check
    std_rgb = float(np.mean(arr.std(axis=(0, 1))))
    if std_rgb < 12.0:
        return False, "The uploaded image lacks anatomical texture or contrast (blank or solid color detected). Please upload a valid retinal scan."

    # 8. Retinal color spectrum check
    # Retinal fundus tissue is dominated by red/orange reflectance; strong blue dominance indicates non-retinal imagery
    r, g, b = mean_rgb[0], mean_rgb[1], mean_rgb[2]
    if b > r * 1.35 and b > 140:
        return False, "The color profile does not match retinal anatomy (unnatural blue dominance detected). Please upload an authentic eye fundus photograph."

    return True, "Valid retinal fundus photograph."


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


def predict_with_details(image: Image.Image, model=None, device=None, validate: bool = True):
    """
    Performs inference on a PIL image and returns all prediction details
    and preprocessed artifacts needed for Grad-CAM.
    
    Args:
        image (PIL.Image.Image): Input retinal fundus image.
        model (torch.nn.Module, optional): Loaded model. If None, loaded via model_loader.
        device (torch.device, optional): Target device.
        validate (bool): Whether to validate if image is an authentic retinal fundus scan.
        
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

    if validate:
        is_valid, validation_msg = validate_fundus_image(image)
        if not is_valid:
            raise ValueError(validation_msg)

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
