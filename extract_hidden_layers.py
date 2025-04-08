import os
import argparse
import torch
import warnings

from model.model import ESM_model
from esm.sdk.api import ESMProtein
from esm.utils.structure.protein_chain import ProteinChain
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_protein(pdb, chain):
    """
    Loads a protein either using a 4-letter PDB code (from RCSB) or a file path.
    """
    if len(pdb) == 4:
        protein_chain = ProteinChain.from_rcsb(pdb.upper(), chain_id=chain)
        protein = ESMProtein.from_protein_chain(protein_chain)
    else:
        if not os.path.isfile(pdb):
            raise FileNotFoundError(f"{pdb} does not exist.")
        if os.path.getsize(pdb) == 0:
            raise ValueError(f"{pdb} is empty.")
        pdb_id = os.path.basename(pdb)
        protein_chain = ProteinChain.from_pdb(pdb, chain_id=chain, id=pdb_id)
        protein = ESMProtein.from_protein_chain(protein_chain)
    return protein

def process_protein(protein, model, use_pdb_seq=True, override_sequence=None):
    """
    Processes the protein to obtain inputs for the backbone model.
    Returns:
        seq_input: Tensor of token ids (shape: [1, L])
        struct_input: Tensor with structure embeddings (shape: [1, L, hidden_size])
        mask: Boolean tensor marking valid tokens
    """
    encoder = model.model.encode(protein)
    # Remove special tokens from beginning/end.
    struct_input = encoder.structure[1:-1].unsqueeze(0)
    seq_input = encoder.sequence[1:-1].unsqueeze(0)
    
    if not use_pdb_seq and override_sequence:
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        token_seq = tokenizer.encode(override_sequence, add_special_tokens=False, return_tensors='np')
        seq_input = torch.from_numpy(token_seq).to(DEVICE)
    
    # Create a mask for valid tokens (assuming token id 4099 is special)
    mask = seq_input != 4099
    return seq_input.to(DEVICE), struct_input.to(DEVICE), mask.to(DEVICE)

def extract_classifier_hidden_layer(model, seq_input, struct_input, mask, desired_layer=10):
    """
    Runs the protein through the ESM-based model and extracts the classifier’s hidden layers.
    Returns the hidden state from the specified layer (squeezed to remove the batch dimension).
    """
    # Run the backbone (assumes method='esm3')
    _, esm_hidden_states = model.model(
        sequence_tokens=seq_input, 
        structure_tokens=struct_input, 
        sequence_id=mask
    )
    # Run the classifier part (which is a transformer) and get its hidden states.
    _, classifier_hidden_states = model.classifier(
        esm_hidden_states[model.layer],
        torch.logical_not(mask),
        return_hidden_states=True
    )
    
    if len(classifier_hidden_states) < desired_layer:
        raise ValueError(f"Classifier only has {len(classifier_hidden_states)} layers; cannot extract layer {desired_layer}.")
    # Squeeze to remove the batch dimension so that shape is [L, d]
    return classifier_hidden_states[desired_layer - 1].squeeze(0)

def init_model(weights_path, method='esm3', layer=22, nheads=6, nlayers=12, dropout=0.1):
    """
    Initializes and returns the ESM_model with the given parameters.
    """
    model = ESM_model(method=method, nlayers=nlayers, nheads=nheads, dropout=dropout, layer=layer).to(DEVICE)
    weights = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(weights, strict=False)
    model.eval()
    return model

def main(args):
    # Read protein references from the input file (one per line)
    with open(args.input_file, "r") as f:
        proteins = [line.strip() for line in f if line.strip()]
    
    # Initialize the model once
    model = init_model(args.weights_path)
    
    all_hidden_layers = []  # List to accumulate each protein's classifier hidden layer tensor
    for pdb_ref in proteins:
        try:
            protein = load_protein(pdb_ref, args.chain)
            seq_input, struct_input, mask = process_protein(protein, model, args.use_pdb_seq, args.sequence)
            hidden_layer = extract_classifier_hidden_layer(model, seq_input, struct_input, mask, desired_layer=args.layer)
            print(f"Successfully processed protein '{pdb_ref}' with hidden layer shape {hidden_layer.shape}")
            all_hidden_layers.append(hidden_layer)
        except Exception as e:
            print(f"Error processing protein '{pdb_ref}': {str(e)}")
    
    # Concatenate all token embeddings along the token dimension
    if all_hidden_layers:
        final_tensor = torch.cat(all_hidden_layers, dim=0)
    else:
        final_tensor = torch.tensor([])
    
    # Save the final concatenated tensor
    out_file = os.path.join(args.out_folder, f"glued_classifier_hidden_layer_{args.layer}.pt")
    torch.save(final_tensor.cpu(), out_file)
    print(f"Final glued tensor saved to '{out_file}' with shape {final_tensor.shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Sequentially extract classifier hidden layers from a file containing protein references."
    )
    parser.add_argument('--input_file', type=str, required=True,
                        help="Path to the file containing protein references (one per line)")
    parser.add_argument('--chain', type=str, default='A',
                        help="Chain ID to use (default: A). For multiple chains, adapt the code accordingly.")
    parser.add_argument('--use_pdb_seq', action='store_true',
                        help="If set, use the sequence extracted from the pdb file")
    parser.add_argument('--sequence', type=str, default=None,
                        help="Optional override sequence")
    parser.add_argument('--out_folder', type=str, default="hidden_layers",
                        help="Folder in which to save the final concatenated tensor")
    parser.add_argument('--layer', type=int, default=10,
                        help="Which classifier hidden layer to extract (default: 10)")
    parser.add_argument('--weights_path', type=str, default="model/weights/dyna1.pt",
                        help="Path to the model weights file")
    args = parser.parse_args()

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
    
    main(args)
