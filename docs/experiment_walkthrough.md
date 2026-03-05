# Experiment Walkthrough

## Dataset: FashionMNIST

### Overview

FashionMNIST is a compact, standardized vision benchmark (60,000 train / 10,000 test) with 10 classes of 28×28 grayscale images, which makes runs fast and comparisons reproducible.
Its scale is large enough to show meaningful convergence/stability differences, yet small enough to support multiple seeds and sensitivity sweeps without excessive compute.

### Augmentation

For the main SGD vs QuickProp comparison, we **avoid or keep augmentation very weak** because data augmentation can be viewed as introducing additional stochasticity by optimizing a sequence of time‑varying proxy losses, which increases gradient variance across steps.
Since QuickProp estimates per-parameter curvature via a secant-like denominator $$g_{t-1}-g_t$$, extra gradient noise from random transforms can destabilize curvature estimates and trigger more fallback/step-limiting events, confounding an optimizer-only comparison.

## Base optimizer: SGD

SGD is the canonical first‑order baseline because each update uses an unbiased stochastic gradient estimate, trading exactness for efficiency—ideal as a reference point when evaluating a curvature‑aware method like QuickProp. Using SGD (optionally with momentum) provides a well-understood control for convergence speed and stability under mini-batch noise.

## Models

### Overall architecture: All-Conv GAP-CNN

The network is all-convolutional with stride‑2 convolutions replacing pooling (simpler downsampling without max-pool) and a 1×1 conv + global average pooling (GAP) head to reduce parameters and improve interpretability/regularization, following established CNN design principles.

### Weights initialization: Xavier (Glorot)

- **Stability:** QuickProp’s secant update ($\Delta w \approx \Delta w_{t-1} \cdot \frac{g_t}{g_{t-1} - g_t}$) is highly sensitive to noisy gradients. Xavier’s conservative variance prevents unstable "exploding" steps in early training.
- **Precision:** Lower initial variance ensures smoother finite-difference curvature estimates, which are critical for the parabolic approximation.
- **Efficiency:** A single, robust scheme minimizes divergence risks and simplifies the 3-day implementation timeline.

### Activation: ReLU and SiLU

For activation, **ReLU** is the standard default, but **SiLU/Swish** $$\mathrm{silu}(x)=x\,\sigma(x)$$ is smooth and often empirically competitive; the smoother gradient can reduce erratic per-step gradient changes that destabilize QuickProp’s secant denominator $$g_{t-1}-g_t$$.
