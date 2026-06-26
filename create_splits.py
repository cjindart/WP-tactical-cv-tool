import os
import shutil
import random

images_dir = 'datasets/game_1_test/boxed_frames/images'
labels_dir = 'datasets/game_1_test/boxed_frames/labels'

# create split folders
for split in ['train', 'val']:
    os.makedirs(f'datasets/game_1_test/boxed_frames/images/{split}', exist_ok=True)
    os.makedirs(f'datasets/game_1_test/boxed_frames/labels/{split}', exist_ok=True)

# get all images
images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
random.shuffle(images)

split_idx = int(len(images) * 0.8)
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

for img in train_imgs:
    shutil.move(os.path.join(images_dir, img), f'datasets/game_1_test/boxed_frames/images/train/{img}')
    label = img.replace('.jpg', '.txt')
    shutil.move(os.path.join(labels_dir, label), f'datasets/game_1_test/boxed_frames/labels/train/{label}')

for img in val_imgs:
    shutil.move(os.path.join(images_dir, img), f'datasets/game_1_test/boxed_frames/images/val/{img}')
    label = img.replace('.jpg', '.txt')
    shutil.move(os.path.join(labels_dir, label), f'datasets/game_1_test/boxed_frames/labels/val/{label}')

print(f'Train: {len(train_imgs)}, Val: {len(val_imgs)}')