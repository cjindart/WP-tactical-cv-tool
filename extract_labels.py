"""
Water polo label extractor
Parses a Sportscode XML timeline and extracts labeled frames
Helper function to take timestamp and return label
"""

import xml.etree.ElementTree as ET
import subprocess
import os

"""
parse_instances(file_path: str) -> tuple[list[dict], list[str]]:
Parses a passed in XML script of timestamps and labels
returns a dictionary of instances for relevant CODES
"""
def parse_instances(file_path: str) -> tuple[list[dict], list[str]]:
    instances = []
    codes = []
    with open(file_path, "r", encoding="utf-16") as f:
        content = f.read()
        root = ET.fromstring(content)

        # get CODES for labels
        for instance in root.iter("instance"):
            CODE = instance.findtext("code")
            if CODE.endswith("6v5") or CODE.endswith("Offense") or CODE.endswith("Transition O"):
                if CODE not in codes:
                    codes.append(CODE)

        def priority(code):
            if code.endswith("6v5"):
                return 0
            elif code.endswith("Transition O"):
                return 1
            return 2

        codes = sorted(codes, key=priority)

        # get all instances with CODES
        for instance in root.iter("instance"):
            CODE = instance.findtext("code")
            if CODE in codes:
                instances.append({
                    "code": CODE,
                    "start": float(instance.findtext("start")),
                    "end": float(instance.findtext("end")),
                })

    return instances, codes

"""
get_label(time: float, instances: list[dict]) -> str:
Gets the specific label for a timestamp
takes highest priority CODE if multiple possible
If no applicable CODE, marked as N/A (timeout, dead time)
"""
def get_label(time: float, instances: list[dict], codes: list[str]) -> str:
    labels = []
    for instance in instances:
        if instance["start"] <= time <= instance["end"]:
            labels.append(instance["code"])

    for code in codes:
        if code in labels:
            return code
    return "N/A"


def main():
    instances, codes = parse_instances("datasets/game_1_test/game_1_test.xml")
    time = input("Input a time!: ")
    label = get_label(float(time), instances, codes)
    print(label)



if __name__ == "__main__":
    main()