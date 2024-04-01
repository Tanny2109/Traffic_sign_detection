from moviepy.editor import ImageSequenceClip
import os

def combine_images_to_video(input_folder, output_path, fps):
    # List all PNG files in the input folder
    image_files = [os.path.join(input_folder, file) for file in sorted(os.listdir(input_folder)) if file.endswith('.png')]
    
    # Create a video clip from the PNG images
    clip = ImageSequenceClip(image_files, fps=fps)
    
    # Write the video file
    clip.write_videofile(output_path)

# Example usage
combine_images_to_video('/home/GTL/tsutar/Traffic_sign_detection/datasets/LISA/vid0/frameAnnotations-vid_cmp2.avi_annotations/', '/home/GTL/tsutar/Traffic_sign_detection/datasets/LISA_vids/output_video_0.mp4', 1)  # Adjust parameters as needed
