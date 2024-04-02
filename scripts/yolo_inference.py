import os
from ultralytics import YOLO
import cv2
import math
from roboflow import Roboflow
import supervision as sv
import cv2
import numpy as np

# import torch
img = "/home/GTL/tsutar/Traffic_sign_detection/datasets/LISA_yolo/train/images/addedLane_1323813414-avi_image0_png_jpg.rf.89b6a75ffb1237a10cfbf952fc4589e7.jpg"
# img = cv2.imread(img)
out_path = '../output/'

rf = Roboflow(api_key="azodSTTVgBufbgTDaoab")
project = rf.workspace().project("road-detection-segmentation")
model2 = project.version(7).model

result2 = model2.predict(img, confidence=10).json() 

labels = [item["class"] for item in result2["predictions"]]

detections = sv.Detections.from_inference(result2)  
mask = detections.mask

image = cv2.imread(img)

polygons = [sv.mask_to_polygons(m) for m in mask]
for polygon in polygons:
        for p in polygon:
                for i in range(len(p)):
                        cv2.line(image, tuple(p[i]), tuple(p[(i+1)%len(p)]), (0, 255, 0), 1)



# Fit horizontal lines through the polygon
left_boundary = []
for polygon in polygons:
        for p in polygon:
                min_x = min(xyxy[0] for xyxy in p)
                left_boundary += [xyxy for xyxy in p if xyxy[0] == min_x]
                # cv2.line(image, (leftmost_point[0], y), (rightmost_point[0], y), (0, 0, 255), 2)
print(left_boundary)
cv2.imwrite(os.path.join(out_path, 'annotated.jpg'), image)

polygon_annotator = sv.PolygonAnnotator()
annotated_image = polygon_annotator.annotate(
    scene = image.copy(),
    detections = detections,
)

# cv2.imwrite(os.path.join(out_path, 'annotated.jpg'), annotated_image)

# annotated_frame = bounding_box_annotator.annotate(
#     scene=image.copy(),
#     detections=detections
# )
# cv2.imwrite(os.path.join(out_path,'annotated.jpg'),annotated_frame)

# annotated_image = mask_annotator.annotate(
#     scene=image, detections=detections)
# annotated_image = label_annotator.annotate(
#     scene=annotated_image, detections=detections, labels=labels)
# cv2.imwrite(os.path.join(out_path,'annotated.jpg'),annotated_image)


# # Load a model
# model = YOLO(
#     "/home/GPU/tsutar/home_gtl/Traffic_sign_detection/scripts/runs/detect/train3/weights/last.pt"
# )
# # print(torch.cuda.devices())

# results = model.predict(
#     img, stream=True, verbose=False, device="cpu"
# )

# for r in results:
#     for b in r.boxes:
#         x1, y1, x2, y2 = b.xyxy[0]
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#         w, h = x2 - x1, y2 - y1
#         start_pt = (x1, y1)
#         end_pt = (x2, y2)
#         cv2.rectangle(img, start_pt, end_pt, color=(255, 0, 0))
        

#         cv2.imwrite(
#             "/home/GPU/tsutar/home_gtl/Traffic_sign_detection/scripts/detected.jpg", img
#         )

