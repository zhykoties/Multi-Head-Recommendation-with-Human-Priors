import torch
from torch import nn, Tensor


class ResBlock(nn.Module):
    """
    A Residual Block module from https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/medusa_model.py.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size, use_norm=False, zero_init=True):
        super().__init__()
        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        if zero_init:                       # default keeps old behaviour
            torch.nn.init.zeros_(self.linear.weight)
        else:                               # small-std trunc-normal for deeper stacks
            torch.nn.init.trunc_normal_(self.linear.weight, std=0.02)
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        if self.use_norm:
            x = self.norm(x)
        return x + self.act(self.linear(x))



class Rescale(nn.Module):
    """
    A PyTorch module for rescaling a tensor by element wise multiplication.
    """

    def __init__(self, size: int, learnable: bool = False) -> None:
        """
        Initializes the Rescale module.

        Args:
            size (int): The size of the multiplication tensor.
            learnable (bool, optional): Whether the multiplication tensor is a learnable parameter. Defaults to False.
        """
        super().__init__()
        if learnable:
            # Initialize the multiplication tensor as a learnable parameter
            # Add a small stdev to avoid all the same initial values
            # TODO: check if 0.1 is a good stdev for initialization
            self.mul_weight = nn.Parameter(torch.ones(size) + 0.1 * torch.randn(size))
        else:
            # Register the multiplication tensor as a buffer (non-trainable)
            self.register_buffer(
                "mul_weight", torch.ones(size) + 0.1 * torch.randn(size)
            )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.mul_weight
