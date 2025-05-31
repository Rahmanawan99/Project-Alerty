from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8n.pt")
result = model("./Test/bus.jpg", show = True , save= True)

cv2.waitKey(0)