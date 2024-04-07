import os
from ultralytics import YOLO
import cv2
import math
from roboflow import Roboflow
import supervision as sv
import cv2
import numpy as np
from scipy.spatial import ConvexHull

# import torch
img = "../output/vid_frames/frame_0000.jpg"
out_path = '../output/'

def bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy        
    points.append((x, y))
    return points

def find_boundary(img_path, rf_api, out_path):
        rf_api = "AutNOr0Ryqh0SP0Wzi24"
        rf = Roboflow(api_key=rf_api)
        project = rf.workspace().project("road-detection-segmentation")
        rf_model = project.version(7).model
        rf_result = rf_model.predict(img_path, confidence=5).json() 
        detections = sv.Detections.from_inference(rf_result)  
        mask = detections.mask
        if mask is None:
                print("no road found")
                return [(0, 0),(0, 0)], [(0, 0),(0, 0)]

        image = cv2.imread(img)
        polygons = [sv.mask_to_polygons(m) for m in mask]
        polygon_annotator = sv.PolygonAnnotator()
        annotated_frame = polygon_annotator.annotate(
                scene=image.copy(),
                detections=detections
        )

        for polygon in polygons:
                for p in polygon:
                        if len(p) < 40:
                                continue
                        points = np.array(p)

                        # Find the leftmost and rightmost points
                        leftmost_point = tuple(points[np.argmin(points[:,0])])
                        rightmost_point = tuple(points[np.argmax(points[:,0])])

                        center_x = (leftmost_point[0] + rightmost_point[0]) / 2

                        # Split the points into left and right groups
                        left_points = points[points[:,0] < center_x]
                        right_points = points[points[:,0] > center_x]

                        # Find the topmost point in each group
                        topmost_point_left = tuple(left_points[left_points[:,1].argmin()])
                        topmost_point_right = tuple(right_points[right_points[:,1].argmin()])

                        ##Draw vertical lines for left and right boundaries
                        cv2.line(image, leftmost_point, topmost_point_left, (0, 255, 255), 1)
                        cv2.line(image, rightmost_point, topmost_point_right, (0, 255, 255), 1)

                        left_boundary_points = bresenham_line(leftmost_point[0], leftmost_point[1], topmost_point_left[0], topmost_point_left[1])
                        right_boundary_points = bresenham_line(rightmost_point[0], rightmost_point[1], topmost_point_right[0], topmost_point_right[1])
        

                        cv2.imwrite(os.path.join(out_path, 'annotated.jpg'), annotated_frame)
                        print("save successfull")

                        if not left_boundary_points or not right_boundary_points:
                                return [(0, 0),(0, 0)], [(0, 0),(0, 0)]
                        
                        return left_boundary_points, right_boundary_points
                return [(0, 0),(0, 0)], [(0, 0),(0, 0)]
        
if __name__ == "__main__":
        left_boundary_points, right_boundary_points = find_boundary(img, "AutNOr0Ryqh0SP0Wzi24", out_path)
        print("Left boundary points: ", left_boundary_points)
        print("Right boundary points: ", right_boundary_points)
        print("Done")
        # find_boundary(img, "AutNOr0Ryqh0SP0Wzi24", out_path)
        # print("Done")
