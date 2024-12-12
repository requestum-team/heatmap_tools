import numpy as np
from typing import Union, Tuple
from hm_tools.utils import calc_dists

    
def generate_hm_dots(dim: Union[int, Tuple[int, int]],
                     coords: np.ndarray,
                     sigma: float = 0.03,
                     offset_vector_layers:bool = False,
                     offset_treshold: float = 0.1, 
                     ) -> np.ndarray:
    """
    Generates a Gaussian heatmap for the given coordinates.

    Parameters:
    coords (np.ndarray): array of [[x, y],] realtive points coords.
    image_shape (tuple): Shape of the output heatmap (height, width).
    sigma (float): Standard deviation for the Gaussian kernel in relative.

    Returns:
    np.ndarray: Heatmap of the given shape with Gaussian blobs at the specified coordinates.
    """

    if isinstance(dim, int):
        dim: Tuple[int, int] = (dim, dim)
    
    sigma_ = sigma * dim[0]
    coords_ = coords * np.array(dim)
    coords_rounder = coords_.astype(int)
   
    height, width = dim
    heatmap = np.zeros(dim)
    
    # Create a grid of (x, y) coordinates
    y, x = np.mgrid[0:height, 0:width]
    
    x = x[None, :, :]  
    y = y[None, :, :]  
    
    # cx, cy = coords_rounder[:, 0][:, None, None],coords_rounder[:, 1][:, None, None]
    cx, cy = coords_rounder[:, 0][:, None, None],coords_rounder[:, 1][:, None, None]
    
    gaussian = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma_**2))
    
    heatmap = np.sum(gaussian, axis=0)
    
    # Normalize heatmap values to be between 0 and 1
    heatmap = np.clip(heatmap, 0, 1)
    
    if offset_vector_layers:
        
        pixels_coords = np.argwhere(heatmap > offset_treshold) 
        
        dists, d_idxs = calc_dists(pixels_coords[:,[1,0]], coords_ )         

        offset_vectors = np.zeros(dim + (2,))

        normalized_components = (coords_[d_idxs][:,[1,0]] - pixels_coords - 0.5)  / np.array(dim)

        offset_vectors[pixels_coords[:, 0], pixels_coords[:, 1],0] = normalized_components[:, 0]
        offset_vectors[pixels_coords[:, 0], pixels_coords[:, 1],1] = normalized_components[:, 1]
        
        heatmap = np.stack([heatmap,offset_vectors[...,0],offset_vectors[...,1]],
                           axis=-1)
       
    return heatmap
