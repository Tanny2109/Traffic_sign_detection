import cv2
from ultralytics import YOLO
import math
import numpy as np
from collections import defaultdict
import supervision as sv
from roboflow import Roboflow
from argparse import ArgumentParser
from moviepy.editor import VideoFileClip
import os

# Extract the coordinates of the bounding box
# x, y, w, h = boxes[0]

# # Calculate the width of the bounding box
# bbox_width = w

# # Calculate the distance from the camera to the object
# distance = (bbox_width * FOCAL_LENGTH) / (2 * math.tan(math.radians(FIELD_OF_VIEW / 2)))

# # Calculate the road width
# road_width = bbox_width * distance * PIXEL_SIZE

class TSDetection:
    def __init__(self, yolo_weight, roboflow_api_key, output_path, input_video_path, output_video_name, vid_frame_save_path):
        # Camera parameters
        self.FOCAL_LENGTH = 27 / 25.4  # 27 mm to inches
        self.PIXEL_SIZE = 1.4e-6 * 39.37  # 1.4 um to inches
        self.FIELD_OF_VIEW = 77.0
        self.sensor_size =  0.393701
        self.yolo_model = YOLO(yolo_weight)
        self.output_path = output_path
        self.rf = Roboflow(api_key=roboflow_api_key)
        self.project = self.rf.workspace().project("road-detection-segmentation")
        self.rf_model = self.project.version(7).model
        # Set tracking parameters
        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.5   # IOU threshold
        self.input_video_path = input_video_path
        self.output_video_name = output_video_name
        self.track_history = defaultdict(lambda: [])
        self.vid_frame_save_path = vid_frame_save_path
    
    def bresenham_line(self,x0, y0, x1, y1):
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

    def find_road(self, img_path):
        print(img_path)
        rf_result = self.rf_model.predict(img_path, confidence=5).json() 
        detections = sv.Detections.from_inference(rf_result)  
        mask = detections.mask
        if mask is None:
            print("no road found")
            return [(0, 0),(0, 0)], [(0, 0),(0, 0)]

        polygons = [sv.mask_to_polygons(m) for m in mask]
        # polygon_annotator = sv.PolygonAnnotator()
        # annotated_frame = polygon_annotator.annotate(
        #     scene=image.copy(),
        #     detections = detections
        # )

        image = cv2.imread(img_path)

        for polygon in polygons:
                for p in polygon:
                        if len(p) < 50:
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
                        cv2.line(image, leftmost_point, topmost_point_left, (0, 255, 0), 2)
                        cv2.line(image, rightmost_point, topmost_point_right, (0, 255, 0), 2)

                        left_boundary_points = self.bresenham_line(leftmost_point[0], leftmost_point[1], topmost_point_left[0], topmost_point_left[1])
                        right_boundary_points = self.bresenham_line(rightmost_point[0], rightmost_point[1], topmost_point_right[0], topmost_point_right[1])

                        # cv2.imwrite(os.path.join(out_path, 'annotated.jpg'), image)
                        if not left_boundary_points or not right_boundary_points:
                                return [(0, 0),(0, 0)], [(0, 0),(0, 0)]

                        return left_boundary_points, right_boundary_points
                
                return [(0, 0),(0, 0)], [(0, 0),(0, 0)]

    def clip_video(self, input_path, output_path, duration):
        # Load the video
        video = VideoFileClip(input_path)
            
        # Clip the video to the specified duration
        clipped_video = video.subclip(0, duration)
            
        # Write the clipped video to the output file
        clipped_video.write_videofile(output_path, codec="mpeg4")
            
        # Close the video file
        video.close()
        os.remove(input_path)
    
    def find_closest(self, points, target):
        return min(points, key=lambda t: abs(t[0] - target[0]))
    
    def estimate_dims(self, pts_on_left, pts_on_right, w, h):
        # distance = (2 * 3.14 * 180) / (w+h * 360) * 1000 + 3 # distance in inches
        ## resize image to original size.
        road_width_in_px = np.sqrt((pts_on_left[0] - pts_on_right[0])**2 + (pts_on_left[1] - pts_on_right[1])**2)
        scale = (40 * 12)/road_width_in_px
        real_w = w*scale
        real_h = h*scale
        return real_w, real_h

    def track(self):
        cap = cv2.VideoCapture(self.input_video_path)
        size = (int(cap.get(3)), int(cap.get(4)))
        vid_writer = cv2.VideoWriter(
            self.output_path + "/"+str(self.output_video_name), cv2.VideoWriter_fourcc(*'mjpg'), 4, size
        )

        counter=0
        # reading the video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if counter > 100:
                 cap.release()

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

                ## save the frames do it just once
                ## cv2.imwrite(self.vid_frame_save_path + "frame_{:04d}.jpg".format(counter), annotated_frame)
                left_boundary, right_boundary = self.find_road(self.vid_frame_save_path + "frame_{:04d}.jpg".format(counter))
                closest_points_on_left = self.find_closest(left_boundary, (x, y))
                closest_points_on_right = self.find_closest(right_boundary, (x, y))

                cv2.line(annotated_frame, closest_points_on_left, closest_points_on_right, (0, 255, 0), 2)

                real_w, real_h = self.estimate_dims(closest_points_on_left, closest_points_on_right, w, h)
                text_x = int(x - w // 2)
                text_y = int(y + h // 2)

                # Draw the text
                cv2.putText(annotated_frame, "Height: {:.1f} in, Width: {:.2f} in".format(real_h,real_w), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                counter+=1

            vid_writer.write(annotated_frame)

        # Release the video capture object and close the display window
        cap.release()
        vid_writer.release()

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
        "--output-video-name", type=str, default="ts_dims_road.mp4"
    )
    parser.add_argument(
        "--roboflow-api-key", type=str, default="CBTptDD9807neChO809i"
    )
    parser.add_argument(
        "--vid-frame-save-path", type=str, default="../output/vid_frames/"
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
        vid_frame_save_path=args.vid_frame_save_path,
    )
    tsd.track()
    # tsd.clip_video(args.output_path + "/"+args.output_video_name, args.output_path + "/final.mp4", 3*60)
