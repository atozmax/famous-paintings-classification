import io
import os
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from app.labels import ARTISTS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "models" / "resnet152_progressive.pth"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Same validation geometry used while the checkpoint was trained.
TRANSFORM = transforms.Compose(
    [
        transforms.Resize(248),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PainterModel:
    def __init__(self, checkpoint=None):
        self.checkpoint = Path(checkpoint or os.environ.get("MODEL_PATH", DEFAULT_CHECKPOINT))
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint}. "
                "Place resnet152_progressive.pth in models/."
            )
        self.device = pick_device()
        self.model = models.resnet152(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(ARTISTS))
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image: Image.Image, top_k: int = 5):
        top_k = max(1, min(top_k, len(ARTISTS)))
        tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probabilities = self.model(tensor).softmax(dim=1)[0]
        scores, indices = torch.topk(probabilities, k=top_k)
        top = [
            {
                "artist": ARTISTS[int(index)],
                "artist_id": int(index),
                "confidence": round(float(score), 6),
            }
            for score, index in zip(scores.tolist(), indices.tolist())
        ]
        return top[0], top

    def predict_bytes(self, payload: bytes, top_k: int = 5):
        image = Image.open(io.BytesIO(payload))
        return self.predict(image, top_k=top_k)
