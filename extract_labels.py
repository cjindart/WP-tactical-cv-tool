"""
Water polo label extractor
Parses a Sportscode XML timeline and extracts labeled frames
Helper function to take timestamp and return label
"""

import xml.etree.ElementTree as ET
import subprocess
import os

XML_PATH = "dataset/game_1.xml"
VIDEO_PATH = "dataset/your_video.mp4"
OUTPUT_PATH = "dataset/frames"
INTERVAL = 5

CODES = [
    "California Offense",
    "California Transition O",
    "Stanford Offense",
    "Stanford Transition O",
]

"""
parse_instances(file_path: str) -> list[dict]
Parses a passed in XML script of timestamps and labels
returns a dictionary of instances for relevant CODES
"""
def parse_instances(file_path: str) -> list[dict]:
    instances = []
    with open(file_path, "r", encoding="utf-16") as f:
        content = f.read()
        root = ET.fromstring(content)
        for instance in root.iter("instance"):
            # print(instance.findtext("code"))
            # print(instance.findtext("start"))
            # print(instance.findtext("end"))
            code = instance.findtext("code")
            if code in CODES:
                # print(code, instance.findtext("start"), instance.findtext("end"))
                instances.append({
                    "code": code,
                    "start": float(instance.findtext("start")),
                    "end": float(instance.findtext("end")),
                })
    # print(instances)
    return instances

"""
get_label(time: float, instances: list[dict]) -> str:
Gets the specific label for a timestamp
takes highest priority CODE if multiple possible
If no applicable CODE, marked as N/A (timeout, dead time)
"""
def get_label(time: float, instances: list[dict]) -> str:
    labels = []
    for instance in instances:
        if instance["start"] <= time <= instance["end"]:
            labels.append(instance["code"])

    for CODE in CODES:
        if CODE in labels:
            return CODE
    return "N/A"


def main():
    instances = parse_instances("datasets/xmls/game_1.xml")
    time = input("Input a time!: ")
    label = get_label(float(time), instances)
    print(label)



if __name__ == "__main__":
    main()