"""R3D-18 (Kinetics-400 pretrained) with Polynomial Adapters.

Backbone is fully frozen; only adapter parameters and the classification
head are trained.  Call get_1x_lr_params() / get_10x_lr_params() to obtain
the two parameter groups used by the optimizer.
"""

import torch
import torch.nn as nn
import torchvision

from network.adapters.poly_adapter import inject_adapters


class R3DAdapted(nn.Module):

    def __init__(
        self,
        num_classes: int,
        Q: int = 4,
        conv_type: str = "laguerre",
        N_lag: int | None = None,
        adapter_stages: tuple = (1, 2, 3, 4),
        bottleneck_rank: int | None = None,
        adapter_mode: str = "poly",
    ):
        super().__init__()

        bb = torchvision.models.video.r3d_18(weights="DEFAULT")
        for p in bb.parameters():
            p.requires_grad_(False)

        self.stem = bb.stem

        layers = [bb.layer1, bb.layer2, bb.layer3, bb.layer4]
        for i, layer in enumerate(layers):
            if (i + 1) in adapter_stages:
                layers[i] = inject_adapters(layer, Q, conv_type, N_lag, bottleneck_rank, adapter_mode)

        self.layer1, self.layer2, self.layer3, self.layer4 = layers
        self.avgpool = bb.avgpool

        self.fc = nn.Linear(512, num_classes)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(x.flatten(1))

    def get_1x_lr_params(self):
        """Adapter parameters (excludes frozen backbone and fc head)."""
        for name, p in self.named_parameters():
            if p.requires_grad and not name.startswith("fc."):
                yield p

    def get_10x_lr_params(self):
        """Classification head parameters."""
        return iter(self.fc.parameters())
