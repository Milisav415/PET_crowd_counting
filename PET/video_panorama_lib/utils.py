# video_panorama_lib/utils.py
import cv2


def extract_frames(video_path, frame_interval=30):
    """
    Extract frames from a video file at a given interval.

    :param video_path: Path to the video file.
    :param frame_interval: Extract one frame every 'frame_interval' frames.
    :return: List of extracted frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frames.append(frame)
        count += 1

    cap.release()
    return frames


def stitch_frames(frames):
    """
    Stitch a list of frames into a panorama.

    :param frames: List of image frames.
    :return: The stitched panorama image or None if stitching fails.
    """
    # Create a stitcher object (for OpenCV 4.x)
    stitcher = cv2.Stitcher_create()
    status, panorama = stitcher.stitch(frames)

    if status != cv2.Stitcher_OK:
        print("Error during stitching. Error code:", status)
        return None

    return panorama
