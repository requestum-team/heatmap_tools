import numpy as np
from typing import Optional
import cv2

def draw_heatmap_with_offsets(
        image: np.ndarray,
        heatmap: np.ndarray,
        offset_x: np.ndarray = None,
        offset_y: np.ndarray = None,
        colormap: Optional[int] = None,
        draw_offset_arrows=False,
        draw_offset_conf=0.5,
        arrows_color=(0, 255, 0)) -> np.ndarray:
    
    height, width = image.shape[:2]
    # heatmap_height, heatmap_width  = heatmap.shape[:2]
    heatmap_height, heatmap_width = heatmap.shape[:2]
    dx = np.zeros([heatmap_height, heatmap_width]) if offset_x is None else offset_x * width
    dy = np.zeros([heatmap_height, heatmap_width]) if offset_y is None else offset_y * height
    tmp = np.clip(heatmap * 255.0, 0.0, 255.0).astype("uint8")
    
    
    cm = cv2.COLORMAP_INFERNO if colormap is None else colormap

    colormap_image = cv2.applyColorMap(tmp, cm)
    
    # colormap_image = cv2.resize(colormap_image, (height, width), interpolation=cv2.INTER_NEAREST)
    colormap_image = cv2.resize(colormap_image, (width, height), interpolation=cv2.INTER_NEAREST)
    
    
    result = image.copy()    
    result = cv2.addWeighted(image.copy(), 0.5, colormap_image, 0.5, 0)

    if draw_offset_arrows and any([offset_x is not None, offset_y is not None]):
        for i in range(heatmap_width):
            for j in range(heatmap_height):
                if heatmap[j, i] < draw_offset_conf:
                    continue
                x1 = (i + 0.5) / heatmap_width * width
                y1 = (j + 0.5) / heatmap_height * height
                
                # x1 = (i) / heatmap_width * width
                # y1 = (j) / heatmap_height * height
                x2 = x1 + dx[j, i]
                y2 = y1 + dy[j, i]
                
                cv2.circle(result, (round(x1), round(y1)), 2, arrows_color, -1)
                cv2.line(result, (round(x1), round(y1)), (round(x2), round(y2)), arrows_color, 1)
                cv2.circle(result, (round(x2), round(y2)), 2, (0,0,255), -1)
    
    return result
