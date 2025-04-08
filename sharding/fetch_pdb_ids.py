import requests

def fetch_all_pdb_ids():
    """
    Fetch the complete list of current PDB IDs using the Holdings API.
    
    Returns:
        list: A list of all current PDB ID strings if successful, or an empty list.
    """
    # Endpoint for retrieving all current PDB IDs.
    # (This endpoint returns a JSON array with the PDB IDs.)
    url = "https://data.rcsb.org/rest/v1/holdings/current/entry_ids"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # The API returns a JSON list of current PDB IDs.
            pdb_ids = response.json()
            return pdb_ids
        else:
            print(f"Error: received HTTP {response.status_code}")
            return []
    except requests.RequestException as e:
        print("Error during API request:", e)
        return []

def fetch_n_pdb_ids(n):
    """
    Get the first n PDB IDs from the complete holdings list.

    Parameters:
        n (int): The number of PDB IDs to extract.

    Returns:
        list: A list containing the first n PDB IDs.
    """
    all_ids = fetch_all_pdb_ids()
    if all_ids:
        # Return only the first n IDs if available.
        return all_ids[:n]
    else:
        return []

def save_ids_to_file(ids, filename="pdb_ids.txt"):
    """
    Save a list of PDB IDs to a text file, one per line.

    Parameters:
        ids (list): List of PDB ID strings.
        filename (str): Name of the output file (default "pdb_ids.txt").
    """
    try:
        with open(filename, "w") as file:
            for pdb_id in ids:
                file.write(pdb_id + "\n")
        print(f"Saved {len(ids)} PDB IDs to {filename}")
    except Exception as e:
        print("Error writing to file:", e)

def main():
    try:
        n = int(input("Enter the number of PDB IDs to fetch: "))
    except ValueError:
        print("Invalid input. Please enter an integer.")
        return

    pdb_ids = fetch_n_pdb_ids(n)
    if pdb_ids:
        save_ids_to_file(pdb_ids)
    else:
        print("No PDB IDs were fetched.")

if __name__ == "__main__":
    main()