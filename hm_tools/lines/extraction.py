import numpy as np
import cv2

def extract_skeleton(image: np.ndarray,
                     thinning_type=None) -> np.ndarray:

    """ Shapes skeleton extraction

    Args:
        image (np.ndarray): image with shapes
        thinning_type (_type_, optional): Defaults to None. cv2.ximgproc.THINNING_GUOHALL may be used

    Returns:
        thinned (np.ndarray): image with skeleton. 
    """
    
    img = image.copy()
    
    if np.max(img) <= 1:
         img = (img * 255).astype(np.uint8)

    img = img.astype(np.uint8)
    
    thinned = cv2.ximgproc.thinning(img,
                                    thinningType=thinning_type)
    return thinned
