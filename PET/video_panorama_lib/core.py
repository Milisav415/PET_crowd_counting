# video_panorama_lib/core.py
import cv2
from .utils import extract_frames, stitch_frames


def create_panorama(video_path, frame_interval=30, output=None):
    """
    Create a panorama image from a video file.

    :param video_path: Path to the input video.
    :param frame_interval: Interval at which frames are extracted.
    :param output: (Optional) If provided, the panorama is saved to this file.
    :return: The panorama image as a numpy array.
    :raises ValueError: If not enough frames are extracted.
    :raises RuntimeError: If stitching fails.
    """
    frames = extract_frames(video_path, frame_interval=frame_interval)
    if len(frames) < 2:
        raise ValueError("Not enough frames extracted to perform stitching!")

    panorama = stitch_frames(frames)
    if panorama is None:
        raise RuntimeError("Failed to create panorama from frames.")

    if output:
        cv2.imwrite(output, panorama)

    return panorama
