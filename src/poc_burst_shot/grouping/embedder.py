from PIL import Image
from torchvision import transforms
import torch
import timm
import numpy as np

# DINOv2 ViT-S/14 (Apache-2.0). Self-supervised features separate "same shot" from "same kind
# of scene" far better than ImageNet classifiers or CLIP-style models; see eval/README.md.
DEFAULT_MODEL = "vit_small_patch14_dinov2.lvd142m"

# ViTs whose timm default input size differs from the 224px images the loader produces;
# timm resamples the position embeddings at load time.
_MODEL_KWARGS = {
    "vit_small_patch14_dinov2.lvd142m": {"img_size": 224},
    "vit_base_patch14_dinov2.lvd142m": {"img_size": 224},
    "vit_small_patch14_reg4_dinov2.lvd142m": {"img_size": 224},
    "vit_base_patch14_reg4_dinov2.lvd142m": {"img_size": 224},
}


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None, batch_size: int = 32):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        # Remove classifier head using num_classes = 0
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0, **_MODEL_KWARGS.get(model_name, {}))
        self.model.eval()
        self.model.to(self.device)

        # tensor conversion + the normalisation the weights were trained with
        cfg = self.model.pretrained_cfg
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg["mean"], std=cfg["std"])
        ])

    def embed(self, image: Image.Image) -> np.ndarray:
        return self.embed_batch([image])[0]

    def embed_batch(self, images: list[Image.Image]) -> np.ndarray:
        """Embed a list of PIL images. Returns L2-normalised vectors of shape (N, D)."""
        chunks = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start:start + self.batch_size]
            tensors = torch.stack([self.transform(img) for img in batch]).to(self.device)
            with torch.no_grad():
                chunks.append(self.model(tensors).cpu().numpy())

        vecs = np.concatenate(chunks)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms
