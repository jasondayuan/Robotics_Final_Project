import cv2
import os
import glob
import re

def create_video_from_images(image_folder, video_name, fps=10):

    image_files = glob.glob(os.path.join(image_folder, 'step_*.png'))
    
    if not image_files:
        print(f"No images found in '{image_folder}' with the pattern 'step_*.png'.")
        return

    def get_step_number(filename):
        match = re.search(r'step_(\d+).png', filename)
        return int(match.group(1)) if match else -1

    image_files.sort(key=get_step_number)

    first_frame = cv2.imread(image_files[0])
    if first_frame is None:
        print(f"Error reading the first image: {image_files[0]}")
        return
        
    height, width, layers = first_frame.shape
    frame_size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_name, fourcc, fps, frame_size)

    print(f"Found {len(image_files)} images. Creating video '{video_name}'...")

    for filename in image_files:
        img = cv2.imread(filename)
        out.write(img)

    out.release()
    print("Video created successfully!")

if __name__ == '__main__':

    IMAGE_DIRECTORY = 'data' 

    OUTPUT_VIDEO_NAME = 'simulation_video.mp4'

    FRAMES_PER_SECOND = 60

    create_video_from_images(IMAGE_DIRECTORY, OUTPUT_VIDEO_NAME, FRAMES_PER_SECOND)