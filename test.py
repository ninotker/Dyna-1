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

def process_protein(protein, use_pdb_seq=True, override_sequence=None):
    """
    Processes the protein to obtain the inputs for the backbone model.
    Returns:
        seq_input: Tensor of token ids (shape: [1, L])
        struct_input: Tensor with structure embeddings (shape: [1, L, hidden_size])
        mask: Boolean tensor marking valid tokens
    """
    encoder = model.model.encode(protein)
    # Remove the special tokens (if any) from beginning/end.
    struct_input = encoder.structure[1:-1].unsqueeze(0)
    seq_input = encoder.sequence[1:-1].unsqueeze(0)
    
    if not use_pdb_seq and override_sequence:
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        token_seq = tokenizer.encode(override_sequence, add_special_tokens=False, return_tensors='np')
        seq_input = torch.from_numpy(token_seq).to(DEVICE)
    
    # Create a mask for tokens; here token id 4099 is treated specially.
    mask = seq_input != 4099
    return seq_input.to(DEVICE), struct_input.to(DEVICE), mask.to(DEVICE)

def extract_classifier_hidden_layer(model, seq_input, struct_input, mask, desired_layer=10):
    """
    Runs the protein through the ESM-based model and extracts the classifier’s hidden layers.
    Since the classifier (a Transformer) is called with return_hidden_states=True,
    it returns a tuple: (final_output, [hidden_state per encoder layer]).
    We extract the hidden state at index desired_layer-1 (if desired_layer is 10, index 9).
    """
    # This example assumes the method is 'esm3' (adjust if using esm2 or other)
    _, esm_hidden_states = model.model(
        sequence_tokens=seq_input, 
        structure_tokens=struct_input, 
        sequence_id=mask
    )
    # Pass the backbone hidden state (using model.layer to index into esm_hidden_states)
    # to the classifier and ask for its hidden states.
    _, classifier_hidden_states = model.classifier(
        esm_hidden_states[model.layer], 
        torch.logical_not(mask), 
        return_hidden_states=True
    )
    
    # Check that there are enough layers
    if len(classifier_hidden_states) < desired_layer:
        raise ValueError(f"Classifier only has {len(classifier_hidden_states)} layers; cannot extract layer {desired_layer}.")
    
    # Return the desired hidden layer; desired_layer=10 corresponds to index 9.
    return classifier_hidden_states[desired_layer - 1]

def main(args):
    # Read protein references from the input file (one reference per line)
    with open(args.input_file, "r") as f:
        proteins_list = [line.strip() for line in f if line.strip()]
    
    # Create the output folder if it does not exist
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
    
    # Initialize the model (using esm3 method as in your current setup)
    global model
    model = ESM_model(method='esm3').to(DEVICE)
    weights = torch.load('model/weights/dyna1.pt', map_location=DEVICE)
    model.load_state_dict(weights, strict=False)
    model.eval()
    
    for pdb_ref in proteins_list:
        try:
            protein = load_protein(pdb_ref, args.chain)
            seq_input, struct_input, mask = process_protein(protein, 
                                                            use_pdb_seq=args.use_pdb_seq, 
                                                            override_sequence=args.sequence)
            # Extract the desired classifier hidden layer (default 10th)
            hidden_layer = extract_classifier_hidden_layer(model, seq_input, struct_input, mask, desired_layer=args.layer)
            
            # Save the hidden layer tensor to a file
            file_stem = os.path.basename(pdb_ref)
            out_filename = f"{file_stem}_classifier_hidden_layer_{args.layer}.pt"
            out_path = os.path.join(args.out_folder, out_filename)
            torch.save(hidden_layer.cpu(), out_path)
            print(f"Saved classifier hidden layer (layer {args.layer}) for protein '{pdb_ref}' to {out_path}")
        except Exception as e:
            print(f"Error processing protein '{pdb_ref}': {str(e)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Batch extract classifier hidden layers from multiple proteins."
    )
    parser.add_argument('--input_file', type=str, required=True,
                        help="Path to text file containing PDB codes/paths (one per line)")
    parser.add_argument('--chain', type=str, default='A',
                        help="Chain ID to use (default: A)")
    parser.add_argument('--use_pdb_seq', action='store_true',
                        help="If set, uses the sequence extracted from the pdb file")
    parser.add_argument('--sequence', type=str, default=None,
                        help="Optional override sequence")
    parser.add_argument('--out_folder', type=str, default="hidden_layers",
                        help="Folder in which to save the classifier hidden layer tensors")
    # Here, the '--layer' argument corresponds to which hidden layer of the classifier you wish to extract (e.g., 10 for the 10th layer).
    parser.add_argument('--layer', type=int, default=10,
                        help="Which classifier hidden layer to extract (default: 10)")
    args = parser.parse_args()
    main(args)
