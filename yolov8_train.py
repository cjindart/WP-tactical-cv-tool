from ultralytics import YOLO

model = YOLO('yolov8x.pt')
model.train(
    data='dataset.yaml',
    epochs=50,
    imgsz=1280,
    batch=4,
    project='runs/train',
    name='waterpolo'
)