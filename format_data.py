"""
Wrapper file to easily run formatting scripts for new data
update game_list with games to format and run
`python format_data.py`

THERE MUST BE: 
- datasets/game_name directory
- datasets/game_name/game_name.xml
- datasets/game_name/game_name.mp4
"""

from extract_frames import *

def run_formatting(game_name: str):
    xml = f"datasets/{game_name}/{game_name}.xml"
    video = f"datasets/{game_name}/{game_name}.mp4"
    format(game_name, str(xml), str(video))

def main():
    # for future with lots of games, add names to list and loop thorugh it
    game_list = ["game_1_test"]
    for game_name in game_list:
        run_formatting(game_name)


if __name__ == "__main__":
    main()