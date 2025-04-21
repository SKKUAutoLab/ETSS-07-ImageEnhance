import torch

def compute_transmission_map(image, A_global, patch_size=15):
    """
    Estimate the Transmission Map for an image and return it as a 2D array.
    
    Parameters:
    - image: Torch tensor of shape (B, C, H, W) representing the input image.
    - A_global: Torch tensor of shape (1, 3, 1, 1) representing the global atmospheric light.
    - patch_size: Size of the patches to use for computing the transmission map.
    
    Returns:
    - transmission_map: Torch tensor of shape (B, H, W) representing the transmission map for each pixel.
    """
    B, C, H, W = image.shape
    
    # Ensure A_global is broadcastable across the batch and spatial dimensions
    A_global = A_global.view(B, C, 1, 1)
    
    # Initialize the transmission map with zeros
    transmission_map = torch.zeros((B, 1, H, W), dtype=torch.float32, device=image.device)
    
    # Padding to handle edge cases when extracting patches
    pad_size = patch_size // 2
    padded_image = torch.nn.functional.pad(image, (pad_size, pad_size, pad_size, pad_size))
    
    unfolded = torch.nn.functional.unfold(padded_image, (patch_size, patch_size), stride=1).view(B, C, patch_size * patch_size, -1)/A_global

    min_values = unfolded.min(dim=2).values
    min_values = min_values.min(dim=1, keepdim=True).values

    transmission_map = 1 - min_values.view(B, 1, H, W)

    
    return transmission_map