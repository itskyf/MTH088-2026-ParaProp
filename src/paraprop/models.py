from typing import Final

from torch import Tensor, nn


class ParaConv(nn.Module):
    """All-convolutional CNN with a GAP head (image-size agnostic).

    Architecture: conv + SiLU; downsampling via stride-2 convolutions (no pooling);
    the head uses 1x1 convolutions to generate per-class score maps followed by
    global average pooling (GAP) to produce logits.

    Args:
        in_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB).
        num_classes: Number of output classes (logit dimension).
        base_channels: Width multiplier. Default of 16 ensures a compact and
            stable model, optimized for second-order algorithms like QuickProp.

    References:
        Lin et al., 2013. Network In Network (GAP & 1x1 convs). arXiv:1312.4400.
        Springenberg et al., 2014. Striving for Simplicity: The All Convolutional Net.
            arXiv:1412.6806.
        Glorot & Bengio, 2010. Understanding the difficulty of training deep feedforward
            neural networks (Xavier initialization). AISTATS 2010.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        self.in_channels: Final[int] = in_channels
        self.num_classes: Final[int] = num_classes

        # Shared activation (no parameters)
        self.activation = nn.ReLU(inplace=False)

        # Channel widths per stage; base_channels controls overall model size
        c1 = base_channels
        c2 = 2 * base_channels
        c3 = 4 * base_channels

        # 1: keep size, then downsample by stride=2.
        self.conv1 = nn.Conv2d(
            in_channels, c1, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(c1, c1, kernel_size=3, stride=2, padding=1, bias=True)

        # 2: mix features, then downsample by stride=2.
        self.conv3 = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(c2, c2, kernel_size=3, stride=2, padding=1, bias=True)

        # 3: deeper features at the current resolution.
        self.conv5 = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6 = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1, bias=True)

        # Head: per-class score maps (1x1 conv), then global average pooling to logits.
        self.classifier_conv = nn.Conv2d(
            c3, num_classes, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self._init_weights()

    def _init_weights(self) -> None:
        """Initializes Conv2d weights using Xavier normal and zeros biases.

        Side Effects:
            Modifies Conv2d weights and biases in-place.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        # Block 1
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))

        # Block 2
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        # Block 3
        x = self.activation(self.conv5(x))
        x = self.activation(self.conv6(x))

        # Head: class maps -> GAP -> logits
        x = self.classifier_conv(x)
        x = self.gap(x)
        return x.flatten(1)
