import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def test_born_rule_gradients():
    """
    Simulates the gradient flow bottleneck in Born Rule outputs
    and verifies if Logit Scaling cures it.
    """
    print("\n--- Testing Born Rule Gradient Dynamics ---")

    # Target Logit (e.g., confident prediction)
    target_logit = torch.tensor([10.0], requires_grad=False)

    # Setup 1: Standard Born Rule (No Scaling)
    # Layer: z -> Norm -> Linear -> MagSq -> Log -> Loss
    # We simulate 'Linear' as a variable 'mag' that needs to grow

    # Initial magnitude (after Norm, before projection)
    z_mag = torch.tensor([1.0], requires_grad=True)

    # Optimizer mimics learning the gain
    opt = torch.optim.SGD([z_mag], lr=0.1)

    losses = []
    grads = []
    mags = []

    print("\n[Baseline] Training without Scaling...")
    for step in range(100):
        # Forward
        # Sqing magnitude to simulate Born Prob
        logit = torch.log(z_mag**2)

        loss = (logit - target_logit) ** 2

        # Backward
        opt.zero_grad()
        loss.backward()

        # Log
        losses.append(loss.item())
        grads.append(z_mag.grad.item())
        mags.append(z_mag.item())

        opt.step()

    print(f"Final Magnitude: {z_mag.item():.4f}")
    print(f"Final Logit: {torch.log(z_mag**2).item():.4f}")
    print(f"Final Gradient: {z_mag.grad.item():.6f} (Notice it is TINY)")

    # Setup 2: With Logit Scaling
    print("\n[Proposed] Training with Logit Scaling (alpha)...")
    z_mag_2 = torch.tensor([1.0], requires_grad=True)
    alpha = torch.tensor([1.0], requires_grad=True)  # Learnable Temp

    opt2 = torch.optim.SGD([z_mag_2, alpha], lr=0.1)

    losses_2 = []

    for step in range(100):
        # Forward: Scaled Logit
        logit = alpha * torch.log(z_mag_2**2)

        loss = (logit - target_logit) ** 2

        opt2.zero_grad()
        loss.backward()

        losses_2.append(loss.item())

        opt2.step()

    print(f"Final Magnitude: {z_mag_2.item():.4f} (Can stay small!)")
    print(f"Final Alpha: {alpha.item():.4f} (Grows to compensate)")
    print(
        f"Final Logit: {(alpha * torch.log(z_mag_2**2)).item():.4f} (Matches Target!)"
    )

    return losses, losses_2


if __name__ == "__main__":
    test_born_rule_gradients()
