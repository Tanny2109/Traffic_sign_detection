import cv2
import os
import moviepy.editor as moviepy
from moviepy.video.io.VideoFileClip import VideoFileClip


def makeVid():
    image_folder = "/home/GPU/tsutar/home_gtl/Traffic_sign_detection/datasets/LISA_yolo/vid_frames/"
    video_name = "video_ts.mp4"

    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M','P','4','V'), 30, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # cv2.destroyAllWindows()
    video.release()

# def convert_avi_to_mp4(avi_file_path, output_name):
#     clip = moviepy.VideoFileClip(avi_file_path)
#     clip.write_videofile(output_name)
#     os.remove('./video_yolo_valid_5fps.mp4')

def clip_video(input_path, output_path, duration):
    # Load the video
    video = VideoFileClip(input_path)
    
    # Clip the video to the specified duration
    clipped_video = video.subclip(0, duration)
    
    # Write the clipped video to the output file
    clipped_video.write_videofile(output_path, codec="mpeg4")
    
    # Close the video file
    video.close()
    os.remove('./video_ts.mp4')

if __name__ == "__main__":
    makeVid()
    # convert_avi_to_mp4("video_yolo_valid_5fps.avi", "video_yolo_valid_5fps.mp4")

    input_video_path = "video_ts.mp4"
    output_video_path = "../output/video_ts_3_min.mp4"

    # Duration for clipping (5 minutes)
    clip_duration = 3 * 60
    clip_video(input_video_path, output_video_path, clip_duration)

