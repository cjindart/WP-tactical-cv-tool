"""
Water polo frame extractor
Takes functions from extract_labels, and parses game film to label frames
"""

from extract_labels import parse_instances, get_label



def main():
    XML_PATH = "datasets/xmls/game_1.xml"
    VIDEO_PATH = "datasets/videos/game_1.mp4"
    OUTPUT_PATH = "datasets/frames"



if __name__ == "__main__":
    main()