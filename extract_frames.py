"""
Water polo frame extractor
Takes functions from extract_labels, and parses game film to label frames
"""

from extract_labels import *
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
def format(game_name: str, xml_path: str, video_path: str):
    instances, codes = parse_instances(xml_path)

    for code in codes:
        path = f"datasets/{game_name}/frames/{code}"
        os.makedirs(path, exist_ok=True)

    duration = get_video_duration(video_path)

    print("="*100)
    print(f"Creating labeled frames for {video_path}")
    for time in tqdm(range(0, int(duration), 5)):
        label = get_label(time, instances, codes)
        if label != "N/A":
            arglist = ["ffmpeg", "-ss", str(time), "-i", video_path, "-frames:v", "1", f"datasets/{game_name}/frames/{label}/frame_{time:06d}s.jpg"]
            subprocess.run(arglist, capture_output=True, text=True)
    print("Done!")
    print("="*100)


def main():
    dir = input("Paste name of game (game folder needs an XML, MP4 in it): ")  # game_1_test
    xml = f"datasets/{dir}/{dir}.xml"
    video = f"datasets/{dir}/{dir}.mp4"
    format(str(dir), str(xml), str(video))


if __name__ == "__main__":
    main()