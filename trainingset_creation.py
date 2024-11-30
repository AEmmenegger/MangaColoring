import cv2
import os
import glob
import numpy as np
import argparse

def extract_frames(video_dir, frames_dir, interval=0.5):
    # Create the output directory if it doesn't exist
    os.makedirs(frames_dir, exist_ok=True)

    # Get a list of all video files in the video directory
    video_files = glob.glob(os.path.join(video_dir, '*'))

    frame_count = 0  # Counter for naming the frames

    for video_file in video_files:
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Error opening video file: {video_file}")
            continue

        # Get the frames per second (fps) and total frame count of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print(f"Cannot determine FPS for {video_file}. Skipping.")
            cap.release()
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps  # Duration in seconds

        # Time interval between frames (e.g., 0.5 seconds)
        timestamps = [x * interval for x in range(int(duration / interval))]

        for timestamp in timestamps:
            # Set the video position to the specific timestamp
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)  # Convert seconds to milliseconds

            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame at {timestamp} seconds in {video_file}")
                continue

            # Construct the frame filename
            frame_filename = os.path.join(frames_dir, f'frame_{frame_count:06d}.png')

            # Save the frame as a PNG image
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

        cap.release()

    print(f"Frame extraction complete. Total frames extracted: {frame_count}")

def generate_masks(frames_dir, masks_dir, threshold1=100, threshold2=200):
    # Create the output directory if it doesn't exist
    os.makedirs(masks_dir, exist_ok=True)

    # Get a list of all frame images
    frame_files = glob.glob(os.path.join(frames_dir, '*'))

    for frame_file in frame_files:
        # Read the image
        img = cv2.imread(frame_file)
        if img is None:
            print(f"Failed to read image: {frame_file}")
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        #edges = cv2.Canny(gray, threshold1=100, threshold2=200)
        edges = cv2.Canny(gray, threshold1=threshold1, threshold2=threshold2)

        # Create a kernel for dilation and erosion
        kernel = np.ones((3, 3), np.uint8)

        # Perform dilation and erosion three times
        processed_edges = edges.copy()
        for _ in range(3):
            processed_edges = cv2.dilate(processed_edges, kernel, iterations=1)
            processed_edges = cv2.erode(processed_edges, kernel, iterations=1)


        # Save the final edge mask
        # Construct the output filename
        filename = os.path.basename(frame_file)
        mask_filename = os.path.join(masks_dir, filename)

        # Save the edge mask image
        cv2.imwrite(mask_filename, processed_edges)

    print("Mask generation complete. Masks saved in training/masks.")

def main():
    parser = argparse.ArgumentParser(description='Frame Extraction and Mask Generation Script')
    parser.add_argument('--extract-frames', action='store_true', help='Only extract frames from videos')
    parser.add_argument('--generate-masks', action='store_true', help='Only generate masks from frames')
    parser.add_argument('--generate-masks-manga', action='store_true', help='Only generate masks from manga panels')
    threshold1_anime = 100
    threshold2_anime = 200
    threshold1_manga = 500
    threshold2_manga = 600


    args = parser.parse_args()

    # If neither option is specified, do both
    if not args.extract_frames and not args.generate_masks:
        args.extract_frames = True
        args.generate_masks = True

    video_dir = 'training/videos'
    frames_dir = 'training/frames'
    manga_panel_dir = 'mangas/panels'
    masks_dir = 'training/masks'

    if args.extract_frames:
        print("Starting frame extraction...")
        extract_frames(video_dir, frames_dir)

    if args.generate_masks:
        print("Starting mask generation...")
        generate_masks(frames_dir, masks_dir, threshold1=threshold1_anime, threshold2=threshold2_anime)
    if args.generate_masks_manga:
        print("Starting mask generation manga...")
        generate_masks(manga_panel_dir, masks_dir, threshold1=threshold1_manga, threshold2=threshold2_manga)

if __name__ == '__main__':
    main()
