import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformer import *
from esm.models.esm3 import ESM3

from transformers import EsmForTokenClassification

os.environ["TOKENIZERS_PARALLELISM"] = 'False'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AF2_model(nn.Module):
    def __init__(self, 
                 nlayers = 12,
                 nheads = 6,
                 dropout = 0.1,
                 length = 400,
                 hidden_size = 128,
                 res_count = 32):
        """
        Initialization of AF2-pair model given embedding of one layer

        Args:
            length: (int) length of the sequence
            res_count: (int) number of residues to select for attention
            nheads: (int) number of heads for Transformer classifier (default: 6)
            dropout: (float) dropout rate for Transformer classifier (default: 0.1)
            length: (int) size of pair representation
            hidden_size: (int) dimension size of model
            res_count: (int) number of residues to call attention to
        """

        super().__init__()
        self.length = length 
        self.res_count = res_count
        self.hidden_size = hidden_size
        self.in_dim = self.res_count * self.hidden_size
        
        self.sel_v = nn.Parameter(torch.randn(self.hidden_size, res_count) * 0.02) 

        self.classifier = Transformer(self.in_dim, 
                                      nheads = nheads,
                                      nlayers = nlayers,
                                      dropout = dropout)
        self.layernorm = nn.LayerNorm(self.hidden_size)

    def forward(self, inputs, mask, store_attn = False):
        """
        Forward function to evaluate logits for AF2 model
        Args:
            inputs: (batch, L, L, features) expecting AF2 pairwise representation
            store_attn: (bool) whether to store attention weight
        Returns:
            logits of shape (batch, L)
        """
        inputs = self.layernorm(inputs)

        v = torch.matmul(inputs, self.sel_v)
        v = F.softmax(v, dim = 2) 
        
        # Store attention map
        if store_attn:
            self.attention_map = v
        
        selected_values = torch.einsum('...bjx,...bjh->...bhx', inputs, v)
        
        flattened_sv = selected_values.reshape(inputs.shape[0], self.length, -1)

        logits = self.classifier(flattened_sv, torch.logical_not(mask)).squeeze()
        
        return logits

class ESM_model(nn.Module):
    def __init__(self, 
                 method='esm3',
                 nlayers=12,
                 nheads=6,
                 dropout=0.1,
                 layer=22,):
        """
        Initialize the ESM_model.

        This class wraps a pre-trained ESM model (ESM-3, ESMC, or ESM2) and adds a custom 
        Transformer classifier on top. The classifier uses embeddings from a specific hidden 
        layer of the ESM model to predict protein dynamics.

        Args:
            method (str): Specifies which ESM model to use ('esm3', 'esmc', or 'esm2').
            nlayers (int): Number of layers in the Transformer classifier.
            nheads (int): Number of attention heads in the Transformer classifier.
            dropout (float): Dropout rate for the Transformer classifier.
            layer (int): Index of the hidden layer from which to extract embeddings.
        """
        
        # Initialize the base class (nn.Module)
        super(ESM_model, self).__init__()
        
        # Store the method and layer index for later use
        self.method = method
        self.layer = layer
        
        # Load the appropriate pre-trained model based on the method specified
        if 'esm3' in self.method:
            # For ESM-3: load the model with the "esm3_sm_open_v1" checkpoint,
            # move it to the specified device, and cast parameters to float32.
            self.model = ESM3.from_pretrained("esm3_sm_open_v1").to(DEVICE, non_blocking=True).to(torch.float32)
            # Set the number of layers and the hidden size based on the model's transformer blocks
            self.n_layers = len(self.model.transformer.blocks)
            self.hidden_size = self.model.transformer.blocks[0].attn.d_model
        elif 'esmc' in self.method:
            # For ESMC: load the "esmc_300m" pre-trained model similarly.
            self.model = ESMC.from_pretrained("esmc_300m").to(DEVICE, non_blocking=True).to(torch.float32)
            self.n_layers = len(self.model.transformer.blocks)
            self.hidden_size = self.model.transformer.blocks[0].attn.d_model
        elif self.method == 'esm2':
            # For ESM2: load a model for token classification that outputs hidden states
            self.model = EsmForTokenClassification.from_pretrained(
                f"facebook/esm2_t30_150M_UR50D",
                num_labels=1,
                hidden_dropout_prob=0.0,
                output_hidden_states=True,
                return_dict=True,
                output_attentions=True
            )
            # Extract configuration details (number of hidden layers and hidden size)
            self.n_layers = self.model.esm.config.num_hidden_layers
            self.hidden_size = self.model.esm.config.hidden_size
            # Use only the underlying ESM model from the token classification wrapper
            self.model = self.model.esm
        
        # Freeze the parameters of the pre-trained model (except for any classifier parts)
        for k, v in self.model.named_parameters():
            if 'classifier' not in k:
                v.requires_grad = False

        # Initialize a custom Transformer classifier that will process the embeddings 
        # from the specified hidden layer to produce final logits.
        self.classifier = Transformer(
            self.hidden_size, 
            nheads=nheads,
            nlayers=nlayers,
            dropout=dropout,
            layer=layer
        )
    
    def forward(self, input, mask):
        """
        Forward pass of the ESM_model.

        Processes input tokens (and structure tokens for ESM-3) through the pre-trained ESM model, 
        extracts embeddings from a specific hidden layer, and passes them through the Transformer 
        classifier to output prediction logits.

        Args:
            input: Batched input tokens. For ESM-3, this is a tuple of (sequence_tokens, structure_tokens).
                   For ESM2, it is simply the sequence tokens.
            mask: A tensor mask indicating valid (non-padding) tokens.
        
        Returns:
            logits: The prediction logits after processing the extracted hidden states.
        """
        if 'esm3' in self.method:
            # For ESM-3, process the input tokens and structure tokens.
            # The model returns a tuple; here we only need the hidden states.
            _, hidden_states = self.model(
                sequence_tokens=input[0], 
                structure_tokens=input[1], 
                sequence_id=mask
            )
            # Extract the hidden state from the specified layer and pass it through the classifier.
            # The mask is inverted (logical NOT) as expected by the classifier.
            logits, classifier_hidden_states = self.classifier(hidden_states[self.layer], torch.logical_not(mask), return_hidden_states=True)
            logits = logits.squeeze()

        elif 'esmc' in self.method:
            # For ESMC, only sequence tokens are used.
            _, hidden_states = self.model(sequence_tokens=input, sequence_id=mask)   
            logits = self.classifier(hidden_states[self.layer], torch.logical_not(mask)).squeeze()
        elif self.method == 'esm2':
            # For ESM2, process the input tokens with an attention mask.
            esm_output = self.model(input_ids=input, attention_mask=mask)
            # Use the hidden states from the output and pass them through the classifier.
            logits = self.classifier(esm_output['hidden_states'], torch.logical_not(mask)).squeeze()
        
        # Return the computed logits (with extra dimensions removed)
        return logits

class BaselineOHE(nn.Module):
    def __init__(self, 
                 nlayers = 12, 
                 nheads = 6, 
                 dropout = 0.1):
        """
        Initialization of baseline one-hot encoder (OHE) model

        Args:
            nlayers: number of layers for Transformer classifier (default: 12)
            nheads: number of heads for Transformer classifier (default: 6)
            dropout: dropout rate for Transformer classifier (default: 0.1)
        """
        super(BaselineOHE, self).__init__()
        self.hidden_size = 24
        self.classifier = Transformer(self.hidden_size, nheads=nheads, nlayers=nlayers, dropout=dropout)
            
    def forward(self, input, mask):
        """
        Forward function to evaluate logits using baseline one-hot encoder

        Args:
            input: batched sequence input
        Returns:
            logits: evaluated logits from model
        """
        emb = F.one_hot(input, num_classes=self.hidden_size).float() # N, L, emb. 16, 300, 24
        logits = self.classifier(emb, torch.logical_not(mask)).squeeze()
        return logits
