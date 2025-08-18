from ultralytics import YOLO

model = YOLO('runs/train2/weights/best.pt')

res = model.predict('images/car_test.png')


print(res)