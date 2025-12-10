import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ComplexLinear(nn.Module):
    """
    True Complex Linear Layer (Manual Implementation).
    W = A + iB
    z = x + iy
    Wz = (Ax - By) + i(Ay + Bx)
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Real and Imaginary weights
        self.weight_real = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias_real = nn.Parameter(torch.Tensor(out_features))
            self.bias_imag = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialization logic for complex numbers
        # Variance doubles because of (Ax - By), so we need to divide by sqrt(2)
        scale = 1.0 / math.sqrt(2.0)
        nn.init.kaiming_uniform_(self.weight_real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_imag, a=math.sqrt(5))

        with torch.no_grad():
            self.weight_real.mul_(scale)
            self.weight_imag.mul_(scale)

        if self.bias_real is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_real)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_real, -bound, bound)
            nn.init.uniform_(self.bias_imag, -bound, bound)

    def forward(self, input_real, input_imag):
        # (Ax - By)
        out_real = F.linear(input_real, self.weight_real) - F.linear(
            input_imag, self.weight_imag
        )
        # (Ay + Bx)
        out_imag = F.linear(input_real, self.weight_imag) + F.linear(
            input_imag, self.weight_real
        )

        if self.bias_real is not None:
            out_real += self.bias_real
            out_imag += self.bias_imag

        return out_real, out_imag


class ComplexLayerNorm(nn.Module):
    """
    Normalization for Complex Vectors.
    We normalize the Magnitude but preserve relative Phase.
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # Learnable gain/bias for magnitude reshaping
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x_real, x_imag):
        # Calculate Magnitude
        mag_sq = x_real**2 + x_imag**2
        mag = torch.sqrt(mag_sq + 1e-6)

        # Mean and Var of Magnitude
        mean_mag = mag.mean(dim=-1, keepdim=True)
        # Unbiased=False matches LayerNorm default
        var_mag = mag.var(dim=-1, keepdim=True, unbiased=False)
        var_mag = torch.clamp(var_mag, min=1e-4)  # Safety Clamp

        # Normalize Magnitude
        norm_mag = (mag - mean_mag) / torch.sqrt(var_mag + 1e-6)

        # Apply affine transform
        norm_mag = norm_mag * self.weight + self.bias

        # Reconstruct with original Phase
        # New z = NewMag * (z / OldMag)
        # Prevent division by zero
        ratio = norm_mag / (mag + 1e-6)

        out_real = x_real * ratio
        out_imag = x_imag * ratio
        return out_real, out_imag


class ComplexSplitGELU(nn.Module):
    """
    Applies GELU to Real and Imag parts independently.
    """

    def forward(self, x_real, x_imag):
        return F.gelu(x_real), F.gelu(x_imag)
