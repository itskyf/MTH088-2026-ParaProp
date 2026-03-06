from collections.abc import Callable

import torch
from torch import Tensor
from torch.optim import Optimizer


class QuickProp(Optimizer):
    """Implements the QuickProp optimizer (Fahlman, 1988).

    QuickProp is a per-parameter, second-order flavored update that fits a
    parabola to the error-vs-weight curve using two consecutive slopes and the
    last weight step, then jumps to the parabola minimum.

    Let S(t) = dE/dw be the accumulated gradient ("slope") at step t and
    Δw(t-1) be the previous parameter update. The core parabolic jump is:

        Δw(t) = [ S(t) / (S(t-1) - S(t)) ] * Δw(t-1)

    Safeguards from Fahlman (1988):
      - Bootstrap / restart with gradient descent (GD) when Δw(t-1) = 0 or the
        denominator is ~0.
      - Maximum growth factor μ: |Δw(t)| <= μ * |Δw(t-1)|.
      - Add a GD term (-lr * S(t)) only when S(t) keeps the same sign as S(t-1)
        (prevents oscillation when the minimum is crossed).

    Notes:
      - QuickProp assumes reasonably smooth curvature; it can be less stable
        with non-smooth activations (e.g., ReLU). In practice it may still run,
        but be prepared to tune lr/mu or try smoother activations.

    Args:
      params: Iterable of parameters to optimize.
      lr: Gradient descent coefficient ε used for bootstrap and the optional GD
        add-on term.
      mu: Maximum growth factor μ (typical: 1.75 per Fahlman).
      eps: Small constant to avoid division by ~0 in (S(t-1) - S(t)).
      weight_decay: L2 weight decay added to the slope: S <- S + wd * w.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        mu: float = 1.75,
        eps: float = 1e-12,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if mu <= 1.0:
            raise ValueError(f"mu should be > 1 for growth limiting, got {mu}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")

        defaults = {"lr": lr, "mu": mu, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Performs a single optimization step.

        Args:
          closure: Optional closure that re-evaluates the model and returns loss.

        Returns:
          Loss value if closure is provided, otherwise None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = float(closure())

        for group in self.param_groups:
            lr: float = group["lr"]
            mu: float = group["mu"]
            eps: float = group["eps"]
            wd: float = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.detach()
                if grad.is_sparse:
                    raise RuntimeError("QuickProp does not support sparse gradients.")

                # Add weight decay to slope to prevent weight blow-up (Fahlman, 1988)
                if wd != 0.0:
                    grad = grad.add(param, alpha=wd)

                state = self.state[param]
                if len(state) == 0:
                    # prev_grad: S(t-1), prev_step: Δw(t-1)
                    state["prev_grad"] = torch.zeros_like(param)
                    state["prev_step"] = torch.zeros_like(param)

                prev_grad: Tensor = state["prev_grad"]
                prev_step: Tensor = state["prev_step"]

                denom = prev_grad - grad  # S(t-1) - S(t)
                can_qp = (prev_step != 0) & (denom.abs() > eps)

                # Parabolic jump: Δw_qp = Δw(t-1) * S(t) / (S(t-1) - S(t))
                qp_step = torch.zeros_like(grad)
                qp_step = torch.where(can_qp, prev_step * grad / denom, qp_step)

                # GD bootstrap/add-on step: Δw_gd = -lr * S(t)
                gd_step = -lr * grad

                # Case logic (element-wise), matching the paper's discussion:
                # 1) Sign change (crossed min): use quadratic term only
                same_sign = (grad * prev_grad) > 0  # strictly same sign
                crossed = can_qp & (~same_sign)

                # 2) Same sign, slope not decreasing: risky growth, scale by mu
                slope_decreased = grad.abs() < prev_grad.abs()
                risky = can_qp & same_sign & (~slope_decreased)

                # 3) 3) Same sign, slope decreased: use quadratic + GD add-on
                good = can_qp & same_sign & slope_decreased

                step = gd_step.clone()  # default fallback (bootstrap)
                step = torch.where(crossed, qp_step, step)
                step = torch.where(good, qp_step + gd_step, step)
                step = torch.where(risky, mu * prev_step, step)

                # Safeguard: |Δw(t)| <= mu * |prev_step| (Fahlman, 1988)
                cap = mu * prev_step.abs()
                step = torch.where(
                    prev_step != 0, torch.clamp(step, min=-cap, max=cap), step
                )

                # Numerical safety: fallback to GD if NaN/Inf
                step = torch.where(torch.isfinite(step), step, gd_step)

                param.add_(step)

                # Book-keeping for next step.
                state["prev_grad"] = grad.clone()
                state["prev_step"] = step.clone()

        return loss
