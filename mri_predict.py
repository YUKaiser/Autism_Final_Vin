from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - handled at runtime when PyTorch is absent.
    torch = None

from mri_model_def import Better3DCNN


VOLUME_SHAPE = (128, 160, 160)
CLASS_NAMES = ("Control Pattern", "Possible ASD Pattern")


class MriPrediction(TypedDict):
    class_id: int
    label: str
    confidence: float
    probabilities: tuple[float, float]


def load_mri_model(checkpoint_path: str | Path) -> Better3DCNN:
    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required to use the MRI model. Install it with 'pip install torch'."
        )

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"MRI model checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    if not isinstance(state_dict, Mapping):
        raise TypeError("MRI checkpoint must contain a model state dictionary.")

    model = Better3DCNN()
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def validate_mri_volume(volume: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume)
    if volume.shape != VOLUME_SHAPE:
        raise ValueError(
            f"Expected a preprocessed MRI volume with shape {VOLUME_SHAPE}; "
            f"received {volume.shape}."
        )
    if volume.dtype.kind not in "iuf":
        raise ValueError("MRI volume must contain real numeric values.")
    if not np.isfinite(volume).all():
        raise ValueError("MRI volume contains NaN or infinite values.")

    volume = np.ascontiguousarray(volume, dtype=np.float32)
    if not np.isfinite(volume).all():
        raise ValueError("MRI values cannot be represented as float32.")
    return volume


def predict_mri(volume: np.ndarray, model: Better3DCNN) -> MriPrediction:
    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required to use the MRI model. Install it with 'pip install torch'."
        )

    volume = validate_mri_volume(volume)
    device = next(model.parameters()).device
    inputs = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(inputs)
        if logits.shape != (1, len(CLASS_NAMES)):
            raise ValueError(
                f"MRI model returned unexpected output shape {tuple(logits.shape)}."
            )
        probabilities = torch.softmax(logits, dim=1)[0]

    class_id = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[class_id].item())
    return {
        "class_id": class_id,
        "label": CLASS_NAMES[class_id],
        "confidence": confidence,
        "probabilities": tuple(float(value) for value in probabilities.tolist()),
    }
