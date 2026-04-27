# WP-tactical-cv-tool
Computer Vision tool to analyze and classify tactical situations for Water Polo game film

What's working so far:
1. processing of pretagged game film
2. labeling of frames throughout pretagged film for training
  - frame sampled every 5 seconds from video and labeled with code from xml

What needs to be implemented (for training):
3. DBSCAN for cap color assignments and team labeling
4. YOLOv8 for spatial locating of players in the pool
5. Translation of frame to top-down diagram of pool w/ players
6. Training and testing CNN on train and test splits of games

To create training data from pretagged game, run

`python format_data.py`

THERE MUST BE: 
- datasets/game_name directory
- datasets/game_name/game_name.xml
- datasets/game_name/game_name.mp4