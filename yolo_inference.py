from ultralytics import YOLO

model = YOLO('yolov8x')

result = model.track('input_videos/vid.mp4', save = True)

