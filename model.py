import torch
import torch.nn as nn
import timm


def build_model(
    model_name="efficientnet_b0",
    num_classes=7,
    pretrained=True,
    freeze_backbone=True
):
    """
    Builds an EfficientNetB0-based image classification model
    using transfer learning.
    """

    print(
        f"Building model: {model_name} "
        f"(pretrained={pretrained}, "
        f"freeze_backbone={freeze_backbone})"
    )

    # Create EfficientNetB0
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )

    # Freeze backbone for transfer learning
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze only the final classification layer
        classifier = model.get_classifier()

        for param in classifier.parameters():
            param.requires_grad = True

    return model


def unfreeze_backbone(model):
    """
    Unfreezes the complete model for fine-tuning.
    """

    print("Unfreezing model parameters for fine-tuning...")

    for param in model.parameters():
        param.requires_grad = True

    return model