import numpy as np

def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2)/ 2)
    center_y = int((y1 + y2)/ 2)
    return(center_x, center_y)

def measure_distance(p1, p2):
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    distance = np.sqrt(np.sum((p1 - p2) ** 2))
    return float(distance)