from PIL import Image
from torchvision import transforms
import torch
import timm
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Embedding backbones the user can pick from in the UI. All are timm models that
# expose a pooled feature vector when created with num_classes=0, and work with
# the simple 224x224 + ImageNet-normalisation transform below. Keys are the timm
# model names; values are human-readable labels for the dropdown.
AVAILABLE_MODELS: dict[str, str] = {
    "efficientnet_b0": "EfficientNet-B0 (default)",
    "resnet50": "ResNet-50",
    "mobilenetv3_large_100": "MobileNetV3-Large",
    "convnext_tiny": "ConvNeXt-Tiny",
    "densenet121": "DenseNet-121",
    "vit_base_patch16_224": "ViT-Base/16",
}

DEFAULT_MODEL = "efficientnet_b0"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Remove classifier head using num_classes = 0
        self.model = timm.create_model(model_name, pretrained=True, num_classes = 0)
        self.model.eval()
        self.model.to(self.device)

        #tensor conversion + imagenet normalisation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = IMAGENET_MEAN, std = IMAGENET_STD)
        ])

    
    def embed(self, image: Image.Image) -> np.ndarray:

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(tensor)

        vec = features.squeeze(0).cpu().numpy()
        return vec / np.linalg.norm(vec)

    def embed_batch(self, images: list[Image.Image]) -> np.ndarray:
        """Embed a list of PIL images. Returns shape (N, D), where D depends on
        the chosen backbone (e.g. 1280 for EfficientNet-B0)."""
        tensors = torch.stack([self.transform(img) for img in images]).to(self.device)

        with torch.no_grad():
            features = self.model(tensors)  # (N, 1280)

        vecs = features.cpu().numpy()
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms