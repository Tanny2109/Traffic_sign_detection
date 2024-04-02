import cv2
from ultralytics import YOLO
import math
import numpy as np
from collections import defaultdict
import supervision as sv
from roboflow import Roboflow
from argparse import ArgumentParser

# Extract the coordinates of the bounding box
# x, y, w, h = boxes[0]

# # Calculate the width of the bounding box
# bbox_width = w

# # Calculate the distance from the camera to the object
# distance = (bbox_width * FOCAL_LENGTH) / (2 * math.tan(math.radians(FIELD_OF_VIEW / 2)))

# # Calculate the road width
# road_width = bbox_width * distance * PIXEL_SIZE

class TSDetection:
    def __init__(self, yolo_weight, roboflow_api_key, output_path, input_video_path, output_video_name):
        # Camera parameters
        self.FOCAL_LENGTH = 27/1000  # 27 mm to meters
        self.PIXEL_SIZE = 1.4 / 1000000  # 1.4 micrometer to meters
        self.FIELD_OF_VIEW = 77.0
        self.yolo_model = YOLO(yolo_weight)
        self.output_path = output_path
        self.rf = Roboflow(api_key=roboflow_api_key)
        self.project = self.rf.workspace().project("road-detection-segmentation")
        # Set tracking parameters
        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.5   # IOU threshold
        self.input_video_path = input_video_path
        self.output_video_name = output_video_name
        self.track_history = defaultdict(lambda: [])
    

    def find_road(self):
            
        sv.process_video(
            source_path=self.output_path + "/"+self.output_video_name,
            target_path=self.output_path + "/ts_road.mp4",
            callback=self.callback,
        )

    def callback(self, scene: np.ndarray, index: int) -> np.ndarray:
        results = self.rf_model.predict(scene, confidence=10).json()
        detections = sv.Detections.from_ultralytics(results)

        polygon_annotator = sv.PolygonAnnotator()

        annotated_frame = polygon_annotator.annotate(
            scene=scene, detections=detections)

        return annotated_frame

    def track(self):
        cap = cv2.VideoCapture(self.input_video_path)
        size = (int(cap.get(3)), int(cap.get(4)))
        vid_writer = cv2.VideoWriter(
            self.output_path + "/"+str(self.output_video_name), cv2.VideoWriter_fourcc('M','J','P','G'), 1, size
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.yolo_model.track(frame, persist=True, device=1)

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
                track = self.track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=1)

            # Display the annotated frame
            vid_writer.write(annotated_frame)

        # Release the video capture object and close the display window
        cap.release()
        vid_writer.release()

        self.find_road()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--yolo-weight", type=str, default="/home/GPU/tsutar/home_gtl/Traffic_sign_detection/scripts/runs/detect/train2/weights/best.pt"
    )
    parser.add_argument(
        "--output-path", type=str, default="../output"
    )
    parser.add_argument(
        "--input-video-path", type=str, default="../output/video_ts_3_min.mp4"
    )
    parser.add_argument(
        "--output-video-name", type=str, default="ts_detection.mp4"
    )
    parser.add_argument(
        "--roboflow-api-key", type=str, default="azodSTTVgBufbgTDaoab"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    tsd = TSDetection(
        yolo_weight=args.yolo_weight,
        roboflow_api_key=args.roboflow_api_key,
        output_path=args.output_path,
        input_video_path=args.input_video_path,
        output_video_name=args.output_video_name,
    )
    tsd.track()

# # Load YOLOv5 model
# weights = '/home/GPU/tsutar/home_gtl/Traffic_sign_detection/scripts/runs/detect/train2/weights/best.pt'
# model = YOLO(weights)
# rf = Roboflow(api_key="azodSTTVgBufbgTDaoab")
# project = rf.workspace().project("road-detection-segmentation")
# model2 = project.version(7).model
# track_history = defaultdict(lambda: [])

# # Load video
# video_path = '../output/video_5_min.mp4'
# cap = cv2.VideoCapture(video_path)

# save_path = "../output"
# size = (int(cap.get(3)), int(cap.get(4)))

# out_writer = cv2.VideoWriter(
#     save_path + "/yolo_tracking_v2.mp4", cv2.VideoWriter_fourcc('M','J','P','G'), 1, size
# )

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model.track(frame, persist=True, device=1)

#     if results[0].boxes.id == None:
#         continue

#     # Get the boxes and track IDs
#     boxes = results[0].boxes.xywh
#     track_ids = results[0].boxes.id.int().tolist()

#     # Visualize the results on the frame
#     annotated_frame = results[0].plot()

#     # Plot the tracks
#     for box, track_id in zip(boxes, track_ids):
#         x, y, w, h = box
#         track = track_history[track_id]
#         track.append((float(x), float(y)))  # x, y center point
#         if len(track) > 30:  # retain 90 tracks for 90 frames
#             track.pop(0)

#         # Draw the tracking lines
#         points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
#         cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=1)

#     # Display the annotated frame
#     out_writer.write(annotated_frame)

# # Release the video capture object and close the display window
# cap.release()
# out_writer.release()
