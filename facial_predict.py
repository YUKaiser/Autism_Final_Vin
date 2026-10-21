import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(channels),
            )
        layers = [BasicBlock(self.in_channels, channels, stride, downsample)]
        self.in_channels = channels
        layers.extend(BasicBlock(self.in_channels, channels) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return torch.flatten(self.avgpool(x), 1)

    def forward(self, x):
        return self.fc(self.forward_features(x))


class FusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.fc2(F.relu(self.bn1(self.fc1(x))))


@dataclass
class FacialModels:
    autism: ResNet18
    emotion: ResNet18
    fusion: FusionModel
    config: dict
    ood_mean: torch.Tensor
    ood_std: torch.Tensor


def load_facial_models(model_dir):
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "facial_config.json").read_text(encoding="utf-8"))

    autism = ResNet18(num_classes=len(config["autism_classes"]))
    autism.load_state_dict(
        torch.load(
            model_dir / "best_autism_resnet18.pth",
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )
    emotion = ResNet18(num_classes=len(config["emotion_classes"]))
    emotion.load_state_dict(
        torch.load(
            model_dir / "best_emotion_resnet18_final.pth",
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )
    fusion = FusionModel(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_classes=config["num_classes"],
    )
    fusion.load_state_dict(
        torch.load(
            model_dir / "best_fusion_model_v2.pth",
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )
    ood_stats = torch.load(
        model_dir / "ood_stats.pt",
        map_location="cpu",
        weights_only=True,
    )

    for model in (autism, emotion, fusion):
        model.eval()
    return FacialModels(
        autism=autism,
        emotion=emotion,
        fusion=fusion,
        config=config,
        ood_mean=ood_stats["mean"].float(),
        ood_std=ood_stats["std"].float().clamp_min(1e-6),
    )


def _prepare_image(image):
    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        image_array = np.asarray(image)
        if image_array.ndim != 3 or image_array.shape[2] != 3:
            raise ValueError("Facial image must have three RGB color channels.")
        if image_array.dtype != np.uint8:
            if not np.issubdtype(image_array.dtype, np.number):
                raise ValueError("Facial image must contain numeric pixel values.")
            if not np.isfinite(image_array).all():
                raise ValueError("Facial image contains non-finite pixel values.")
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(image_array, mode="RGB")

    pil_image = pil_image.resize((224, 224), Image.Resampling.BILINEAR)
    pixels = np.asarray(pil_image, dtype=np.float32) / 255.0
    pixels = (pixels - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return torch.from_numpy(pixels.transpose(2, 0, 1)).unsqueeze(0)


@torch.inference_mode()
def predict_facial(image, models):
    image_tensor = _prepare_image(image)
    features = models.autism.forward_features(image_tensor)
    autism_logits = models.autism.fc(features)
    emotion_logits = models.emotion(image_tensor)
    emotion_probabilities = torch.softmax(emotion_logits, dim=1)

    fusion_input = torch.cat((features, emotion_probabilities), dim=1)
    if fusion_input.shape[1] != models.config["input_dim"]:
        raise ValueError(
            f"Facial fusion model expected {models.config['input_dim']} features, "
            f"received {fusion_input.shape[1]}."
        )
    logits = models.fusion(fusion_input)
    temperature = float(models.config["optimal_temperature"])
    fusion_probabilities = torch.softmax(logits / temperature, dim=1)[0]
    autism_probabilities = torch.softmax(autism_logits, dim=1)[0]

    autism_class = models.config["autism_classes"].index("autistic")
    emotion_names = models.config["emotion_classes"]
    positive_probability = float(fusion_probabilities[autism_class])
    threshold = float(models.config["optimal_threshold"])
    ood_distance = torch.linalg.vector_norm(
        (features[0] - models.ood_mean) / models.ood_std
    ).item()

    return {
        "class_id": int(positive_probability >= threshold),
        "label": (
            "Possible ASD Pattern"
            if positive_probability >= threshold
            else "Lower ASD Pattern"
        ),
        "asd_probability": positive_probability,
        "threshold": threshold,
        "confidence": float(fusion_probabilities.max()),
        "autism_model_probability": float(autism_probabilities[autism_class]),
        "emotions": {
            name: float(probability)
            for name, probability in zip(
                emotion_names, emotion_probabilities[0].tolist()
            )
        },
        "ood_distance": ood_distance,
        "is_ood": ood_distance > float(models.config["ood_threshold"]),
    }
