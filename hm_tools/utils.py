import numpy as np
import scipy.interpolate as si
import cv2
from scipy.spatial import KDTree


def linear_interpolation(cv: np.ndarray,
                         n: int = 100) -> np.ndarray:
    """
    Calculate n samples on a linear interpolation that passes through the given control vertices.
    
    Args:
        cv (np.ndarray): 2D Array of control vertices (x and y coordinates).
        n (int): Number of samples to return.
    
    Returns:
        res (np.ndarray): Interpolated coordinates as a 2D array.
    """
    
    x, y = cv[:, 0], cv[:, 1]

    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    y_sorted = y[sorted_idx]
    
    unique_x, unique_indices = np.unique(x_sorted, return_index=True)
    y_sorted = y_sorted[unique_indices] 
    
    interp_func = si.interp1d(unique_x, y_sorted, kind='linear')
    
    xfit = np.linspace(np.min(unique_x), np.max(unique_x), n)
    
    yfit = interp_func(xfit)
    
    res = np.vstack([xfit, yfit]).T
    
    return res

def spline_interpolation(cv: np.ndarray,
                         n: int = 100,
                         k: int = 3) -> np.ndarray:
    """ 
    Calculate n samples on a spline that intersects the given points (cubic).
    Args:
        cv (np.ndarray): 2D Array of control vertices (x and y coordinates)
        n (int): Number of samples to return
        k (int): Degree for the approximation polynom
    
    Returns:
        res (np.ndarray): spline coords 2D array. 
    """

    x, y = cv[:, 0], cv[:, 1]
    is_closed = np.allclose(cv[0], cv[-1])
    
    # TODO not common case
    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    y_sorted = y[sorted_idx]

    unique_x, unique_indices = np.unique(x_sorted, return_index=True)
    y_sorted = y_sorted[unique_indices]  # Retain corresponding y-values
    
    if len(unique_x) <= k:
        k = len(unique_x) - 1 
    
    if is_closed:
        # ellipse-like 
        xfit = np.linspace(0, 1, n)
        tck, u = si.splprep([x, y], s=0, per=True,k=k)
        xfit, yfit = si.splev(xfit, tck)
    else:
        # single spline 
        xfit = np.linspace(np.min(unique_x), np.max(unique_x), n)   
        tck = si.splrep(unique_x, y_sorted, s=0, k=k)
        yfit = si.splev(xfit, tck)

    res = np.vstack([xfit, yfit]).T
    
    return res


def bspline_interpolation(cv: np.ndarray,
                          n: int = 100,
                          degree: int = 3) -> np.ndarray:
    
    """ Calculate n samples for approximating the curve with control vertices (basis spline)
    
    Args:
        cv(np.ndarray): Array ov control vertices
        n(int): Number of samples to return
        degree (int): Curve degree of approximation, simple line interpolation if degree = 1 
        
    Returns:
        res (np.ndarray): bspline coords 2D array. 
    """
    cv = np.asarray(cv)
    count = cv.shape[0]

    degree = np.clip(degree,1,count-1)
    kv = np.array([0]*degree + list(range(count-degree+1)) + [count-degree]*degree,dtype='int')
    u = np.linspace(0,(count-degree),n)
    res = np.array(si.splev(u, (kv,cv.T,degree))).T
    
    return res


def draw_polyline(image: np.ndarray,
                  points: np.ndarray,
                  thickness: int = 3) -> np.ndarray:
    
    """Draws polyline using opencv

    Args:
        image (np.ndarray): Image where will be drawed
        points (np.ndarray): Points coords
        thickness (int): Line thickness. Defaults to 3.

    Returns:
        image: array with drawed image
    """
    cv2.polylines(image, [points], isClosed=False, color=(255, 255, 255), thickness=thickness)
    # image[image>0] = 255
    # cv2.GaussianBlur(image, (0, 0), sigmaX=thickness / 3)
    return image


def calc_dists(pixels_coords: np.ndarray,
               line_coords: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """
    Calculates the Euclidean distances between each pixel and its nearest skeleton point.

    Args:
        pixels_coords (np.ndarray): An 2D array (x, y) of pixel coordinates.
        line_coords (np.ndarray):  An 2D array (x, y) of skeleton lines coordinates.

    Returns:
        dists (np.ndarray): 1D array of distances, where each element is the distance from a pixel to its nearest skeleton point.
        d_idx (np.ndarray): 1D array of indices, where each element is the index of the nearest skeleton point in `line_coords` for the corresponding pixel.
    """
    
    tree = KDTree(line_coords)
    dists, d_idx = tree.query(pixels_coords, k=1)
    return dists, d_idx


def apply_linear_gradient(image: np.ndarray,
                          pixels_coords: np.ndarray,
                          dists: np.ndarray,
                          sigma: float = 0.5) -> np.ndarray:
    
    """Applies linear gradient on the image with shapes based on pixels dists

    Args:
        image (np.ndarray): 2D image gradient to be applied
        pixels_coords (np.ndarray): 2D array of the 
        dists (np.ndarray): 1D array with dists of corresponding pixels coords to the skeleton line
        sigma (float, optional): Param for the gradient strength. Defaults to 0.5.

    Returns:
        image (np.ndarray) : 2D array with applied gradient 
    """

    max_dist = np.max(dists)
    min_dist = np.min(dists)


    if max_dist == min_dist:
        norm_dists = np.ones_like(dists) * 255  
    else:
        # Normalizing distances and apply sigma for gradient control
        norm_dists = 255 * (1 - (dists - min_dist) / (max_dist - min_dist))
        norm_dists = (norm_dists ** sigma).clip(0, 255).astype(np.uint8)
    
    image[pixels_coords[:, 0], pixels_coords[:, 1]] = norm_dists
    
    return image


def apply_gaussian_gradient(image: np.ndarray,
                            pixels_coords: np.ndarray,
                            dists: np.ndarray,
                            sigma: float = 0.5) -> np.ndarray:
    
    """Applies gaussian gradient on the image with shapes based on pixels dists

    Args:
        image (np.ndarray): 2D image gradient to be applied
        pixels_coords (np.ndarray): 2D array of the 
        dists (np.ndarray): 1D array with dists of corresponding pixels coords to the skeleton line
        sigma (float, optional): Param for the gradient strength. Defaults to 0.5.

    Returns:
        image (np.ndarray) : 2D array with applied gradient 
    """
    # max_dist = np.max(dists)
    # min_dist = np.min(dists)

    # if max_dist == min_dist:
    #     norm_dists = np.ones_like(dists) * 255
    # else:
    #     norm_dists = (dists - min_dist) / (max_dist - min_dist)
    
    gaussian_dists = np.exp(-(dists ** 2) / (2 * sigma ** 2))
    
    image[pixels_coords[:, 0], pixels_coords[:, 1]] = gaussian_dists
    
    return image


def line_offset(dim: tuple[int,int],
                dists: np.ndarray,
                d_idxs: np.ndarray,
                skeleton_coords: np.ndarray,
                pixels_coords: np.ndarray,
                treshold: float = 0.1) -> np.ndarray:
     
    """Generates per pixel normalized vectors offset for X and Y axes.

    Args:
        dim (tuple[int,int]): image dimention
        dists (np.ndarray): 1D array with per pixel distances to the skeleton line
        d_idxs (np.ndarray): 1D array with indexes for corresponding skeleton points.
        skeleton_coords (np.ndarray): 2D array with skeleton line (x,y) points. 
        pixels_coords (np.ndarray): 2D array with drawed line pixels coords.
        treshold (float, optional): Heatmap treshold for offset generations, offsets will be generated only for pixels with greater than treshold. Defaults to 0.1.

    Returns:
        offset_vectors: Array of shape (h,w,2) with X and Y vectors offsets values.
    """
    offset_vectors = np.zeros(dim + (2,))
    
    normalized_components = (skeleton_coords[d_idxs] - pixels_coords - 0.5) / np.array(dim)

    offset_vectors[pixels_coords[:, 0], pixels_coords[:, 1],0] = normalized_components[:, 0]
    offset_vectors[pixels_coords[:, 0], pixels_coords[:, 1],1] = normalized_components[:, 1]

    return offset_vectors