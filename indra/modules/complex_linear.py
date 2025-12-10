import torch
import torch.nn as nn
import math


class ComplexLinear(nn.Module):
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
        # Initialization logic for complex weights
        # We use a variation of Kaiming initialization adapted for complex numbers
        nn.init.kaiming_uniform_(self.weight_real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_imag, a=math.sqrt(5))
        if self.bias_real is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_real)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_real, -bound, bound)
            nn.init.uniform_(self.bias_imag, -bound, bound)

    def forward(self, input):
        # Input is expected to be complex tensor (real + 1j * imag)
        # or we could handle separate real/imag parts.
        # Assuming input is complex tensor for cleaner API as per V5 desire.

        if not torch.is_complex(input):
            # Fallback or error? Let's assume user might pass float and we treat it as real
            input_r = input
            input_i = torch.zeros_like(input)
        else:
            input_r = input.real
            input_i = input.imag

        # (a + bi)(c + di) = (ac - bd) + i(ad + bc)
        # W = wr + i*wi
        # X = xr + i*xi
        # WX = (wr*xr - wi*xi) + i(wr*xi + wi*xr)

        real_out = torch.nn.functional.linear(
            input_r, self.weight_real, self.bias_real
        ) - torch.nn.functional.linear(input_i, self.weight_imag, None)

        imag_out = torch.nn.functional.linear(
            input_r, self.weight_imag, None
        ) + torch.nn.functional.linear(input_i, self.weight_real, self.bias_imag)

        return torch.complex(real_out, imag_out)
