from ultralytics import YOLO
import cv2
import math

from roboflow import Roboflow
import supervision as sv
import cv2

# import torch
img = "/home/GTL/tsutar/Traffic_sign_detection/datasets/LISA_yolo/train/images/addedLane_1323813414-avi_image0_png_jpg.rf.89b6a75ffb1237a10cfbf952fc4589e7.jpg"
# img = cv2.imread(img)

rf = Roboflow(api_key="azodSTTVgBufbgTDaoab")
project = rf.workspace().project("road-detection-segmentation")
model2 = project.version(7).model

result2 = model2.predict(img, confidence=40).json() 

labels = [item["class"] for item in result2["predictions"]]

detections = sv.Detections.from_inference(result2)  

label_annotator = sv.LabelAnnotator()
mask_annotator = sv.MaskAnnotator()

image = cv2.imread(img)

annotated_image = mask_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels)

# sv.plot_image(image=annotated_image, size=(16, 16))
cv2.imwrite("annotated.jpg",annotated_image)


# # Load a model
# model = YOLO(
#     "/home/GPU/tsutar/home_gtl/Traffic_sign_detection/scripts/runs/detect/train3/weights/last.pt"
# )
# # print(torch.cuda.devices())

# results = model.predict(
#     img, stream=True, verbose=True, device="cuda"
# )

# for r in results:
#     for b in r.boxes:
#         x1, y1, x2, y2 = b.xyxy[0]
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#         w, h = x2 - x1, y2 - y1
#         cls = b.cls[0]
#         # cvzone.cornerRect(img, (x1, y1, w, h), colorR=colors[int(cls)])
#         start_pt = (x1, y1)
#         end_pt = (x2, y2)
#         cv2.rectangle(img, start_pt, end_pt, color=(255, 0, 0))
#         # conf = math.ceil((box.conf[0] * 100)) / 100

#         # name = classNames[int(cls)]
#         cv2.imwrite("detected.jpg",img)

        # cv2.imwrite(
        #     "/home/GPU/tsutar/home_gtl/Traffic_sign_detection/scripts/detected.jpg", img
        # )

