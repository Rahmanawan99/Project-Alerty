from sqlite3 import Cursor
from flask import app
from sqlalchemy import CursorResult
from ultralytics import YOLO
import cvzone
import cv2
import math
import os
from flask_mysqldb import MySQL
from datetime import datetime
import mysql.connector
from flask import current_app


def video_detection(path_x):

    videocapture = path_x

    cap = cv2.VideoCapture(path_x)
    model = YOLO('best.pt')

    # Reading the classes
    classnames = []
    with open('classes.txt','r') as f:
        classnames = f.read().splitlines()


    # create the frames directory if it doesn't exist
    frames_dir = 'static/frames'
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # set up timer and interval
    timer = 0
    frame_counter = 0
    interval = 10

    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="alerty"
    )
    cur = conn.cursor()

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 240), interpolation=cv2.INTER_AREA)
        result = model(frame,stream=True)

        # Getting bbox,confidence and class names informations to work with
        # Getting bbox,confidence and class names informations to work with
        for info in result:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])
                if confidence > 45:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),1)
                    if Class < len(classnames):
                        cv2.putText(frame, f'{classnames[Class]} {confidence}%', (x1 + 8, y1 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    else:
                        # Handle the case where the index is out of bounds
                        cv2.putText(frame, f'Unknown {confidence}%', (x1 + 8, y1 - 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                    # increment timer
                    timer += 1

                    # print detection name and reset timer
                    if timer == interval:
                        if Class < len(classnames):
                            if classnames[Class] == "No Helmet":
                              # Get the current time and date
                                print(f'Detected object: {classnames[Class]}')
                                # Get the current time and date
                                current_date = datetime.now().strftime("%Y-%m-%d")
                                current_time = datetime.now().strftime("%H:%M:%S")
                                query = "INSERT INTO alerts (Location, Violation, Site, Date, Time) VALUES (%s, %s, %s, %s, %s)"
                                values = ('Room-3', 'Not Wearing Helmet', 'Zone-3', current_date, current_time)
                                cur.execute(query, values)
                                conn.commit()
                                # Save the frame as an image
                                filename = f"{current_date}_{frame_counter}.jpg"
                                cv2.imwrite(os.path.join(frames_dir, filename), frame)
                        timer = 0

        yield frame
        # increment frame counter
        frame_counter += 1
        # cv2.imshow('frame',frame)
        # cv2.waitKey(1)
cv2.destroyAllWindows()
