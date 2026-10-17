import io
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models

from src.data import eval_transform
from src.labels import ARTISTS


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_model(num_classes, checkpoint=None, pretrained=True):
    weights = models.ResNet152_Weights.DEFAULT if pretrained and checkpoint is None else None
    model = models.resnet152(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    return model


class PainterModel:
    def __init__(self, checkpoint):
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint}. "
                "Place resnet152_progressive.pth in models/."
            )
        self.device = pick_device()
        self.transform = eval_transform()
        self.model = create_model(len(ARTISTS), checkpoint=self.checkpoint, pretrained=False)
        self.model.to(self.device)
        self.model.eval()

    def predict_batch(self, images):
        batch = torch.stack([self.transform(image.convert("RGB")) for image in images])
        batch = batch.to(self.device)
        with torch.no_grad():
            probabilities = self.model(batch).softmax(dim=1)
        return probabilities.cpu()

    def predict(self, image, top_k=5):
        top_k = max(1, min(top_k, len(ARTISTS)))
        probabilities = self.predict_batch([image])[0]
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

    def predict_bytes(self, payload, top_k=5):
        image = Image.open(io.BytesIO(payload))
        return self.predict(image, top_k=top_k)
