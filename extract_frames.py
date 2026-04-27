"""
Water polo frame extractor
Takes functions from extract_labels, and parses game film to label frames
"""

from extract_labels import parse_instances, get_label
import subprocess
import os
from tqdm import tqdm

def get_video_duration(file_path) -> float:
    arglist = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path]
    end = subprocess.run(arglist, capture_output=True, text=True)
    return float(end.stdout.strip())

"""
1. Parse the XML to get codes and all labeled instances
2. Get the video duration so you know when to stop
3. Loop through every 5 seconds from 0 to the end of the video
4. For each timestamp, call get_label to find out what's happening
5. Extract that frame from the video using ffmpeg
6. Save it into the correct folder based on the label
"""
def main():
    XML_PATH = "datasets/game_1_test/game_1_test.xml"
    VIDEO_PATH = "datasets/game_1_test/game_1_test.mp4"

    instances, codes = parse_instances(XML_PATH)

    for code in codes:
        path = f"datasets/game_1_test/frames/{code}"
        os.makedirs(path, exist_ok=True)

    duration = get_video_duration(VIDEO_PATH)

    print(f"Creating labeled frames for {VIDEO_PATH}")
    for time in tqdm(range(0, int(duration), 5)):
        label = get_label(time, instances, codes)
        if label != "N/A":
            arglist = ["ffmpeg", "-ss", str(time), "-i", VIDEO_PATH, "-frames:v", "1", f"datasets/game_1_test/frames/{label}/frame_{time:06d}s.jpg"]
            subprocess.run(arglist, capture_output=True, text=True)
    print("Done!")


if __name__ == "__main__":
    main()