from ultralytics import YOLO

model = YOLO('yolov8x.pt')

results = model.predict(
    source='datasets/game_1_test/frames/Stanford Offense/frame_000460s.jpg',
    classes=[0],    
    conf=0.16,
    save=True,       
    name='game_1_test'  
)