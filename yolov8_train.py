"""
Fine-tune YOLOv8 on water polo player detection.

Run AFTER bootstrap_labels.py + Roboflow review. Expects a dataset.yaml exported
from Roboflow in YOLOv8 format, placed at the project root.

Classes (after Roboflow relabeling):
  0: player_dark
  1: player_light
  2: goalkeeper
"""

from ultralytics import YOLO

model = YOLO("yolov8x.pt")
model.train(
    data="dataset.yaml",
    epochs=50,
    imgsz=1280,
    batch=4,
    project="runs/train",
    name="waterpolo",
)
