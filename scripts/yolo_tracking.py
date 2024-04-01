import cv2
from ultralytics import YOLO
import math
import numpy as np
from collections import defaultdict

# Load YOLOv5 model
weights = '/home/GPU/tsutar/home_gtl/Traffic_sign_detection/scripts/runs/detect/train2/weights/best.pt'
model = YOLO(weights)

track_history = defaultdict(lambda: [])

# Set tracking parameters
conf_thres = 0.4  # confidence threshold
iou_thres = 0.5   # IOU threshold

# Load video
video_path = '../output/video_5_min.mp4'
cap = cv2.VideoCapture(video_path)

save_path = "../output"
size = (int(cap.get(3)), int(cap.get(4)))

out_writer = cv2.VideoWriter(
    save_path + "/yolo_tracking_v2.mp4", cv2.VideoWriter_fourcc('M','J','P','G'), 1, size
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.track(frame, persist=True, device=1)

    if results[0].boxes.id == None:
        continue

    # Get the boxes and track IDs
    boxes = results[0].boxes.xywh
    track_ids = results[0].boxes.id.int().tolist()

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Plot the tracks
    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x), float(y)))  # x, y center point
        if len(track) > 30:  # retain 90 tracks for 90 frames
            track.pop(0)

        # Draw the tracking lines
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=3)

    # Display the annotated frame
    out_writer.write(annotated_frame)

# Release the video capture object and close the display window
cap.release()
out_writer.release()