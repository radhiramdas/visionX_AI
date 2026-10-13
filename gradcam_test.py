import os
import sys
import torch
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# =========================
# 1. Configuration
# =========================

MODEL_PATH = "best_model_efficientnet_b0.pth"

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


# =========================
# 2. LetterboxResize (matching dataset.py)
# =========================

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


# =========================
# 3. Load model
# =========================

print("Loading EfficientNetB0...")

model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=7
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

# Handle different checkpoint formats
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict, strict=True)

model.eval()

print("Model loaded successfully.")


# =========================
# 4. Image preprocessing
# =========================

transform = transforms.Compose([
    LetterboxResize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# 5. Select input image
# =========================

if len(sys.argv) > 1:
    IMAGE_PATH = sys.argv[1].strip().strip('"')
else:
    IMAGE_PATH = input(
        "\nEnter the path of a retinal fundus image: "
    ).strip().strip('"')


if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )


# =========================
# 6. Load image
# =========================

original_image = Image.open(IMAGE_PATH).convert("RGB")

# Apply the same LetterboxResize to 224x224
letterbox = LetterboxResize(IMAGE_SIZE)
letterboxed_image = letterbox(original_image)

# Image for visualization (must be the same letterboxed 224x224 image given to the model)
rgb_image = np.array(
    letterboxed_image
).astype(np.float32) / 255.0

# Image for model
input_tensor = transform(original_image).unsqueeze(0)


# =========================
# 7. Prediction
# =========================

with torch.no_grad():
    output = model(input_tensor)

probabilities = torch.softmax(output, dim=1)

predicted_class = torch.argmax(
    probabilities,
    dim=1
).item()

confidence = probabilities[0, predicted_class].item()

print("\nPrediction")
print("-----------------------------")
print("Class:", CLASSES[predicted_class])
print("Confidence:", f"{confidence * 100:.2f}%")
print("-----------------------------")


# =========================
# 8. Grad-CAM
# =========================

print("\nGenerating Grad-CAM...")

target_layers = [model.conv_head]

targets = [
    ClassifierOutputTarget(predicted_class)
]

with GradCAM(
    model=model,
    target_layers=target_layers
) as cam:

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]


# =========================
# 9. Create visualization
# =========================

visualization = show_cam_on_image(
    rgb_image,
    grayscale_cam,
    use_rgb=True
)


# =========================
# 10. Save result
# =========================

output_name = "gradcam_result.jpg"

Image.fromarray(visualization).save(
    output_name
)

print("\nGrad-CAM generated successfully!")
print("Saved as:", output_name)