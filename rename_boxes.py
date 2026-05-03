import os
import re

labels_dir = 'datasets/game_1_test/boxed_frames/labels'
images_dir = 'datasets/game_1_test/boxed_frames/images'

for filename in os.listdir(labels_dir):
    if not filename.endswith('.txt'):
        continue
    # extract the original frame name (e.g. frame_000460s)
    match = re.search(r'(frame_\w+)\.txt', filename)
    if match:
        new_name = match.group(1) + '.txt'
        os.rename(
            os.path.join(labels_dir, filename),
            os.path.join(labels_dir, new_name)
        )
        print(f'{filename} -> {new_name}')

print('Done!')