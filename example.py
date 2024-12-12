from hm_tools.lines.generation import generate_hm_spline
import numpy as np
import cv2


if __name__ == "__main__":
    
    dim = 50
    n = 100
    sigma = 0.01
    image = np.zeros((dim, dim))

    coords = np.array([
        [10, 10],
        [40, 40],
        [15, 12],
        [1, 1]]) / dim

    spline_hm = generate_hm_spline(dim=dim,
                                   control_vertices=coords,
                                   n=n,
                                   sigma=sigma,
                                   offset_vector_layers=False)

    spline_hm = cv2.resize(spline_hm, (640, 640))
    cv2.imshow("", spline_hm)
    cv2.waitKey(0)
    cv2.destroyAllWindows()