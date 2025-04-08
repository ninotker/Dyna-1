import os
import random

def read_pdb_ids(input_file):
    """
    Read all PDB IDs from the specified file.

    Parameters:
        input_file (str): Path to the file containing PDB IDs (one per line).

    Returns:
        List of PDB ID strings.
    """
    try:
        with open(input_file, 'r') as file:
            # Read non-empty, stripped lines
            pdb_ids = [line.strip() for line in file if line.strip()]
        return pdb_ids
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return []

def shard_ids_randomly(pdb_ids, packet_size=1000):
    """
    Randomly split the given list of PDB IDs into packets of a given size.

    The function shuffles the list of PDB IDs randomly (ensuring reproducibility
    by setting a seed optionally) and then slices them into chunks (shards)
    of packet_size elements.

    Parameters:
        pdb_ids (list): List of PDB IDs.
        packet_size (int): Number of IDs per shard.

    Returns:
        A list where each element is a list containing a shard of PDB IDs.
    """
    # Optionally, set a seed for reproducibility
    random.seed(42)
    shuffled_ids = pdb_ids.copy()
    random.shuffle(shuffled_ids)
    
    # Partition the shuffled list into chunks of packet_size
    shards = [shuffled_ids[i:i+packet_size] for i in range(0, len(shuffled_ids), packet_size)]
    return shards

def save_shards(shards, output_folder="pdb_shards"):
    """
    Save each shard into a separate text file within an output folder.

    The files are named shard_1.txt, shard_2.txt, etc.

    Parameters:
        shards (list): A list of shards (each shard is a list of PDB IDs).
        output_folder (str): The folder to create and save shard files.
    """
    # Create the folder if it does not exist.
    os.makedirs(output_folder, exist_ok=True)
    
    for idx, shard in enumerate(shards, start=1):
        output_file = os.path.join(output_folder, f"shard_{idx}.txt")
        try:
            with open(output_file, 'w') as file:
                file.write("\n".join(shard))
            print(f"Saved shard {idx} with {len(shard)} IDs to {output_file}")
        except Exception as e:
            print(f"Error writing to {output_file}: {e}")

def main():
    input_file = "pdb_ids.txt"
    pdb_ids = read_pdb_ids(input_file)
    
    if not pdb_ids:
        print("No PDB IDs found, please check the input file.")
        return

    shards = shard_ids_randomly(pdb_ids, packet_size=1000)
    print(f"Total shards generated: {len(shards)}")
    save_shards(shards)

if __name__ == "__main__":
    main()