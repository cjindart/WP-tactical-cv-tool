# WP-tactical-cv-tool
Computer Vision tool to analyze and classify tactical situations for Water Polo game film

What's working so far:
1. processing of pretagged game film
2. labeling of frames throughout pretagged film for training (frame sampled every 5 seconds from video and labeled with code from xml)

What needs to be implemented (for training):
1. DBSCAN for cap color assignments and team labeling
2. YOLOv8 for spatial locating of players in the pool
3. Translation of frame to top-down diagram of pool w/ players
4. Training and testing CNN on train and test splits of games

To create training data from pretagged game, run

`python format_data.py`

THERE MUST BE: 
- datasets/game_name directory
- datasets/game_name/game_name.xml (xml file of tagged game)
- datasets/game_name/game_name.mp4 (video corresponding to xml file)

Git Organization Note:
- Large files (mp4, pngs) stored in Google Drive: [link here](https://drive.google.com/drive/folders/1jrp1yFUGbhTdEtCArY3w43N5YopzDXa5?usp=drive_link)
 - mp4 = raw video 
 - pngs = labeled frames

Code only in this repo. XML files committed alongside .py files.
