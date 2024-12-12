import numpy as np
from hm_tools.utils import spline_interpolation, draw_polyline, calc_dists, line_offset, apply_gaussian_gradient, linear_interpolation
from typing import Union, Tuple, Optional

  
def generate_hm_spline(dim: Union[int, Tuple[int, int]],
                       control_vertices: np.ndarray,
                       n: int,
                       sigma: float,
                       thickness: Optional[int] = None,
                       degree: int = 3,
                       offset_vector_layers: bool = False,
                       offset_treshold: float = 0.1,
                       ) -> np.ndarray:
    """Generates line heatmap.

    Args:
        dim (Union[int, Tuple[int, int]]): heatmap dimentions
        control_vertices (np.ndarray): 2D array with (x,y) pivot line relative coords 
        n (int): Point сount in the interpolated line. More points provide more accurate lines.
        thickness (int, optional): Heatmap line thickness. If None, will be calculated autmatically
        sigma (float): Heatmap gradient strength. 
        
        degree (int): Polynom degree for spline interpolation.
        If necessary, will be automatically reduced due to an insufficient number of control vertices.

        offset_vector_layers (bool, optional): If True, offset channels will be generated. Defaults to False.
        offset_treshold (float, optional): Offset generation treshold, will be generated only for pixels with greater than treshold. Defaults to 0.1.

    Returns:
        heatmap (np.ndarray): 2D (h, w) array with heatmap, array of shape (h, w, 3) if offset_vector_layers param is True 
    """
    
    if isinstance(dim, int):
        dim: Tuple[int, int] = (dim, dim)
    
    control_vertices_ = control_vertices * np.array(dim)
    sigma_ = sigma * dim[0]
    
    if degree < 2:
        spline_coords = linear_interpolation(cv=control_vertices_,
                                             n=n)
    
    else:
        spline_coords = spline_interpolation(cv=control_vertices_,
                                             n=n)
    
    skeleton_coords = spline_coords[:, [1, 0]]
    heatmap = np.zeros(shape=dim)
    
    if not thickness:
        thickness = int(round(6 * sigma_)) 
    
    
    def rounder(x):
        if (x-int(x) >= 0.5):
            return np.ceil(x)
        else:
            return np.floor(x)

    rounder_vec = np.vectorize(rounder)
    spline_coords_rounder = rounder_vec(spline_coords)
    
    
    draw_polyline(image=heatmap,
                #   points=spline_coords,
                #   points=np.round(spline_coords).astype(int),
                  points=spline_coords_rounder.astype(int),
                  thickness=thickness)       
    
    pixels_coords = np.argwhere(heatmap > 0) 

    
    dists, d_idxs = calc_dists(pixels_coords=pixels_coords+0.5,
                               line_coords=skeleton_coords) 
 
    apply_gaussian_gradient(image=heatmap,
                pixels_coords=pixels_coords,
                dists=dists,
                sigma=sigma_)
    
    if offset_vector_layers:
        offset_vector = line_offset(dim=dim,
                                    dists=dists,
                                    d_idxs=d_idxs,
                                    skeleton_coords=skeleton_coords,
                                    pixels_coords=pixels_coords,
                                    treshold=offset_treshold)
        
        heatmap = np.stack([heatmap,offset_vector[..., 0], offset_vector[..., 1]],
                           axis=-1)

    return heatmap
