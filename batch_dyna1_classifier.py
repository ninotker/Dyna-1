import os
import glob
import argparse
import torch
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    
    # Create mask for valid tokens (treat token id 4099 as special).
    mask = seq_input != 4099
    return seq_input.to(DEVICE), struct_input.to(DEVICE), mask.to(DEVICE)


def extract_classifier_hidden_layer(model, seq_input, struct_input, mask, desired_layer=10):
    """
    Runs the protein through the ESM-based model and extracts the classifier’s hidden layers.
    Returns the hidden state from the specified layer.
    """
    # Assume method is 'esm3'
    _, esm_hidden_states = model.model(
        sequence_tokens=seq_input, 
        structure_tokens=struct_input, 
        sequence_id=mask
    )
    # Pass the backbone hidden state (using model.layer) to the classifier,
    # requesting its hidden states.
    _, classifier_hidden_states = model.classifier(
        esm_hidden_states[model.layer], 
        torch.logical_not(mask), 
        return_hidden_states=True
    )
    
    if len(classifier_hidden_states) < desired_layer:
        raise ValueError(f"Classifier only has {len(classifier_hidden_states)} layers; cannot extract layer {desired_layer}.")
    # Return the desired hidden layer, squeezing the batch dimension.
    # Shape will become [L, d] if input was [1, L, d]
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


def process_shard(shard_file, out_folder, chain, use_pdb_seq, override_sequence, desired_layer, weights_path):
    """
    Processes a single shard (a text file listing protein references).
    For each protein, it extracts the classifier hidden layer tensor,
    then concatenates all token embeddings (across proteins) and saves the result.
    """
    # Each process initializes its own model instance.
    model = init_model(weights_path)
    
    hidden_layers_list = []
    with open(shard_file, "r") as f:
        proteins = [line.strip() for line in f if line.strip()]
    
    for pdb_ref in proteins:
        try:
            protein = load_protein(pdb_ref, chain)
            seq_input, struct_input, mask = process_protein(protein, model, use_pdb_seq, override_sequence)
            hidden_layer = extract_classifier_hidden_layer(model, seq_input, struct_input, mask, desired_layer)
            # hidden_layer shape: [L, d]. Append to our list.
            hidden_layers_list.append(hidden_layer)
            # Print successful processing for this protein.
            print(f"Successfully processed protein '{pdb_ref}' with hidden layer shape {hidden_layer.shape}")
        except Exception as e:
            print(f"Error processing protein '{pdb_ref}' in file '{shard_file}': {str(e)}")
    
    if hidden_layers_list:
        # Concatenate all token embeddings along dim=0; final shape: [N, d] where N = sum(L for all proteins)
        shard_tensor = torch.cat(hidden_layers_list, dim=0)
    else:
        shard_tensor = torch.tensor([])
    
    # Save the concatenated tensor for this shard.
    shard_basename = os.path.splitext(os.path.basename(shard_file))[0]
    out_filename = f"{shard_basename}_classifier_hidden_layer_{desired_layer}.pt"
    out_path = os.path.join(out_folder, out_filename)
    torch.save(shard_tensor.cpu(), out_path)
    print(f"Processed shard '{shard_file}' saved to '{out_path}' with tensor shape {shard_tensor.shape}")
    return out_path


def glue_shards(shard_paths, final_out_path):
    """
    Loads the saved shard tensors and concatenates them into a single final tensor.
    """
    tensor_list = []
    for shard_path in shard_paths:
        try:
            t = torch.load(shard_path)
            if t.numel() > 0:
                tensor_list.append(t)
        except Exception as e:
            print(f"Error loading shard tensor '{shard_path}': {str(e)}")
    if tensor_list:
        final_tensor = torch.cat(tensor_list, dim=0)
    else:
        final_tensor = torch.tensor([])
    
    torch.save(final_tensor.cpu(), final_out_path)
    print(f"Glued final tensor saved to '{final_out_path}' with shape {final_tensor.shape}")
    return final_tensor


def main(args):
    # Get list of shard files (assume .txt files in the input directory)
    shard_files = glob.glob(os.path.join(args.input_dir, "*.txt"))
    if not shard_files:
        raise ValueError(f"No shard files (.txt) found in directory {args.input_dir}")
    
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
    
    weights_path = args.weights_path  # e.g. "model/weights/dyna1.pt"
    
    processed_shard_paths = []
    # Process shards in parallel using a ProcessPoolExecutor.
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                process_shard,
                shard_file,
                args.out_folder,
                args.chain,
                args.use_pdb_seq,
                args.sequence,
                args.layer,
                weights_path
            ): shard_file for shard_file in shard_files
        }
        for future in as_completed(futures):
            shard_file = futures[future]
            try:
                out_path = future.result()
                processed_shard_paths.append(out_path)
            except Exception as e:
                print(f"Error processing shard file '{shard_file}': {str(e)}")
    
    # Optionally, glue the output shard tensors into one big tensor.
    if args.glue:
        final_out_path = os.path.join(args.out_folder, f"glued_classifier_hidden_layer_{args.layer}.pt")
        glue_shards(processed_shard_paths, final_out_path)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(
        description="Parallel processing of shard files to extract and glue classifier hidden layers."
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help="Directory containing shard .txt files (each with protein references, one per line)")
    parser.add_argument('--chain', type=str, default='A',
                        help="Chain ID to use (default: A)")
    parser.add_argument('--use_pdb_seq', action='store_true',
                        help="If set, uses the sequence extracted from the pdb file")
    parser.add_argument('--sequence', type=str, default=None,
                        help="Optional override sequence")
    parser.add_argument('--out_folder', type=str, default="hidden_layers",
                        help="Folder in which to save the classifier hidden layer tensors")
    parser.add_argument('--layer', type=int, default=10,
                        help="Which classifier hidden layer to extract (default: 10)")
    parser.add_argument('--weights_path', type=str, default="model/weights/dyna1.pt",
                        help="Path to the model weights file")
    parser.add_argument('--num_workers', type=int, default=5,
                        help="Number of parallel workers (default: 5)")
    parser.add_argument('--glue', action='store_true',
                        help="If set, glues the processed shard tensors into one final tensor")
    args = parser.parse_args()
    main(args)
