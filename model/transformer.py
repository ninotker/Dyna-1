import math                     # Import math module for mathematical operations
import torch                    # Import PyTorch library for tensor computations
import torch.nn as nn           # Import PyTorch's neural network module with alias 'nn'
from torch.autograd import Variable  # Import Variable (older autograd API; often not needed in newer PyTorch versions)
from torch.nn.utils import weight_norm  # Import weight normalization utility for potential use

# Define a custom Transformer class that extends PyTorch's nn.Module
class Transformer(nn.Module):
    def __init__(self, 
                 in_dim,         # Input dimension size for the embeddings/features
                 nlayers = 12,   # Number of transformer encoder layers (default: 12)
                 nheads = 6,     # Number of attention heads per encoder layer (default: 6)
                 dropout = 0.1,  # Dropout rate to be applied within the transformer layers (default: 0.1)
                 hidden_size = 2048,  # Dimension of the hidden layer in the feedforward network (default: 2048)
                 layer = -1):    # Specific layer to use from the input embeddings (default: -1, typically the final layer)
        super().__init__()          # Initialize the parent nn.Module class
        
        """
        Instantiation of a Transformer classifier

        Args:
            in_dim: (int) input dimension size
            nlayers: (int) number of layers for Transformer classifier (default: 12)
            nheads: (int) number of heads for Transformer classifier (default: 6)
            dropout: (float) dropout rate for Transformer classifier (default: 0.1)
            hidden_size: (int) dimension size of model
            layer: (int) layer to use from model (default: -1)
        """
                
        self.layer = layer  # Store the specified layer index to use (if not -1, select a specific layer from the input)
        
        # Define the first layer normalization with a very small epsilon for numerical stability
        self.norm1 = nn.LayerNorm(in_dim, eps = 1e-12, elementwise_affine = True)
        
        # Define the second layer normalization, also using LayerNorm
        self.norm2 = nn.LayerNorm(in_dim, eps = 1e-12, elementwise_affine = True)
        
        # Create a single Transformer Encoder layer using PyTorch's built-in TransformerEncoderLayer.
        # This layer includes multi-head self-attention and a feedforward network.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = in_dim,          # Set the model dimension to the input dimension
            nhead = nheads,            # Number of attention heads for multi-head attention
            dim_feedforward = hidden_size,  # Dimension of the feedforward network
            dropout = dropout,         # Dropout rate for regularization
            norm_first = True,         # Apply normalization before the attention and feedforward operations (pre-norm)
            batch_first = True         # Use batch-first ordering for input tensors (batch dimension comes first)
        )
        
        # Stack multiple encoder layers to form the full Transformer encoder.
        # The number of layers is defined by the 'nlayers' parameter.
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers = nlayers)
        
        # Define the final output linear layer that maps from the input dimension to a single output value.
        # This is typically used for classification or regression tasks.
        self.output_layer = nn.Linear(in_dim, 1)
        
        # Apply a custom weight initialization function to all submodules of this Transformer.
        self.apply(self._init_weights)

    def forward(self, embedding, mask, return_hidden_states=False):
        x = embedding
        if not torch.is_tensor(embedding) and self.layer != -1:
            x = embedding[self.layer]
        x = self.norm1(x)
        
        hidden_states = []
        # Iterate over each encoder layer in the Transformer encoder
        for layer in self.encoder.layers:
            x = layer(x, src_key_padding_mask=mask)
            hidden_states.append(x)
        
        x = self.norm2(x)
        x = self.output_layer(x) 
        
        if return_hidden_states:
            return x, hidden_states
        return x


    @staticmethod
    def _init_weights(module):
        """
        Custom weight initialization function applied to all modules.
        This function sets biases of LayerNorm and Linear layers to zero,
        and fills the weights of LayerNorm layers with ones.
        """
        # Check if the module is a LayerNorm
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()     # Initialize bias with zeros
            module.weight.data.fill_(1.0)  # Initialize weights with ones
        
        # Check if the module is a Linear layer and has a bias attribute
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()     # Initialize the bias with zeros
