import torch.nn as nn


class EncoderMLP(nn.Module):
    """
    A flexible MLP that creates a network with decreasing hidden dimensions.

    The network architecture is built as follows:
      - Linear interpolation is done between the input_dim and output_dim
      - When used as an encoder, the final layer outputs output_dim (defaults to hidden_channels).
 

    Args:
        input_dim (int): Dimensionality of the input.
        hidden_channels (int): Base hidden dimension.
        num_layers (int): Number of layers used in the encoder (for the decreasing sizes).
        output_dim (int): Output dimension. If None, defaults to hidden_channels.
        dropout (float): Dropout rate applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_channels: int = 32,
        num_layers: int = 3,
        output_dim: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Use output_dim if provided, otherwise default to hidden_channels
        if output_dim is None:
            output_dim = hidden_channels
            
        # Build a list of sizes: starting at input_dim then interpolating to output_dim
        difference = output_dim - input_dim
        delta = difference // num_layers
        
        sizes = [input_dim] + [
            input_dim + delta * (i + 1) for i in range(num_layers - 1)
        ] + [output_dim]
            
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            # For every layer except the last one, add ReLU (and dropout if requested)
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Encoded representation
        """
        return self.model(x)
