import torch

# Load the .pt file
embeds = torch.load("2F3N_classifier_hidden_layer_10.pt")

# If the .pt file is a tensor, print its dimensions:
print(embeds.shape)

# If the file is a dictionary (common in checkpoints), inspect its keys and dimensions:
if isinstance(embeds, dict):
    for key, value in embeds.items():
        if hasattr(value, "shape"):
            print(f"{key}: {value.shape}")
