try:
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - handled at runtime when PyTorch is absent.
    nn = None


if nn is not None:
    class Better3DCNN(nn.Module):
        def __init__(self):
            super().__init__()

            self.features = nn.Sequential(
                nn.Conv3d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm3d(16),
                nn.ReLU(),
                nn.MaxPool3d(2),
                nn.Conv3d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                nn.MaxPool3d(2),
                nn.Conv3d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.MaxPool3d(2),
                nn.Conv3d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d(1),
            )

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(128, 2),
            )

        def forward(self, x):
            return self.classifier(self.features(x))
else:
    class Better3DCNN:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "PyTorch is required to use the MRI model. Install it with 'pip install torch'."
            )
