import os
import uuid
from datetime import datetime
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Predictable directory for saving generated Grad-CAM images
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GENERATED_DIR = os.path.join(BACKEND_DIR, "generated")

# Ensure the generated directory exists automatically
os.makedirs(DEFAULT_GENERATED_DIR, exist_ok=True)


def generate_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    rgb_image: np.ndarray,
    target_class_idx: int,
    output_dir: str = None,
    filename: str = None
) -> str:
    """
    Generates a Grad-CAM visualization for the specified class index,
    matching the implementation verified in gradcam_test.py.
    
    Target layer: model.conv_head
    Targets: ClassifierOutputTarget(target_class_idx)
    Overlay image: rgb_image (must be the SAME letterboxed 224x224 image given to the model)
    
    Args:
        model (torch.nn.Module): Loaded EfficientNetB0 model.
        input_tensor (torch.Tensor): Preprocessed input tensor (shape 1, 3, 224, 224).
        rgb_image (np.ndarray): Letterboxed RGB image as float32 array in [0.0, 1.0] (shape 224, 224, 3).
        target_class_idx (int): The integer class index to explain (predicted class).
        output_dir (str, optional): Target directory for saving visualization. Defaults to backend/generated.
        filename (str, optional): Custom filename. If None, generates a unique filename.
        
    Returns:
        str: Unique filename of the saved Grad-CAM image.
    """
    if output_dir is None:
        output_dir = DEFAULT_GENERATED_DIR

    os.makedirs(output_dir, exist_ok=True)

    # 1. Target the final convolutional feature extractor layer: model.conv_head
    target_layers = [model.conv_head]

    # 2. Target the predicted class index
    targets = [ClassifierOutputTarget(target_class_idx)]

    # 3. Compute Grad-CAM activation map
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )[0]

    # 4. Create heatmap overlay on the exact letterboxed image
    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    # 5. Generate a unique filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        filename = f"gradcam_{timestamp}_{unique_id}.jpg"

    save_path = os.path.join(output_dir, filename)

    # 6. Save visualization image
    Image.fromarray(visualization).save(save_path, format="JPEG", quality=95)
    print(f"[GradCAM] Visualization successfully saved to: {save_path}")

    return filename
