import os
import torch
import timm

# Cached model and device instances
_model = None
_device = None


# Reliable Project Root Resolution
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

# Default to selected Experiment 3 fine-tuned checkpoint
EXPERIMENT_3_CHECKPOINT_REL = os.path.join("tuning_results", "experiment_3_finetuned", "best_model.pth")
DEFAULT_CHECKPOINT_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, EXPERIMENT_3_CHECKPOINT_REL))


def get_default_checkpoint_path():
    """
    Locates the selected Experiment 3 fine-tuned EfficientNetB0 checkpoint.
    Reliably resolves path based on the project root directory.
    """
    if os.path.exists(DEFAULT_CHECKPOINT_PATH):
        return DEFAULT_CHECKPOINT_PATH

    # Fallback to relative path from current working directory
    alt_path = os.path.abspath(EXPERIMENT_3_CHECKPOINT_REL)
    if os.path.exists(alt_path):
        return alt_path

    return DEFAULT_CHECKPOINT_PATH


def load_model(checkpoint_path=None):
    """
    Loads the trained EfficientNetB0 model with 7 output classes using
    the selected Experiment 3 fine-tuned checkpoint.
    
    Requirements:
    - EfficientNetB0 architecture
    - pretrained=False
    - num_classes=7
    - Loads weights from tuning_results/experiment_3_finetuned/best_model.pth
    - Uses strict=True for parameter verification
    - Uses CPU if CUDA is unavailable
    - Sets model to eval() mode for inference
    
    Args:
        checkpoint_path (str, optional): Custom path to the model checkpoint.
        
    Returns:
        tuple: (model: torch.nn.Module, device: torch.device)
    """
    global _model, _device

    # 1. Determine device (CUDA if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ModelLoader] Using device: {device}")

    # 2. Locate checkpoint file reliably using absolute project root path
    if checkpoint_path is None:
        checkpoint_path = get_default_checkpoint_path()

    checkpoint_path = os.path.abspath(checkpoint_path)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. "
            f"Please ensure '{EXPERIMENT_3_CHECKPOINT_REL}' exists in the project root."
        )

    # 3. Print the full checkpoint path when the model loads
    print(f"[ModelLoader] Loading checkpoint from: {checkpoint_path}")

    # 4. Create model architecture (EfficientNetB0 with 7 classes)
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=7
    )

    # 5. Load weights from checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    # Handle various checkpoint formats (raw state_dict or dictionary wrappers)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError(f"Unexpected checkpoint format: {type(checkpoint)}")

    # Verify classifier shape compatibility (7 classes, 1280 features)
    if "classifier.weight" in state_dict:
        classifier_shape = state_dict["classifier.weight"].shape
        if classifier_shape[0] != 7 or classifier_shape[1] != 1280:
            raise ValueError(
                f"Incompatible classifier shape: {list(classifier_shape)}. "
                "Expected 7 output classes and 1280 input features for EfficientNetB0."
            )

    # Load all model weights with strict=True
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    print(f"[ModelLoader] Successfully loaded all weights with strict=True from: {checkpoint_path}")
    print("[ModelLoader] Model set to eval() mode for inference.")

    _model = model
    _device = device
    return _model, _device


def get_model():
    """
    Returns the cached model and device. If not already loaded, loads the model.
    """
    global _model, _device
    if _model is None or _device is None:
        return load_model()
    return _model, _device
