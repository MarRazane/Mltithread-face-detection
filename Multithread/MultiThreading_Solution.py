import cv2
import json
import sqlite3
import threading
from queue import Queue

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    detected_objects = [{"class": "face", "bbox": [x, y, w, h]} for (x, y, w, h) in faces]
    return detected_objects


def save_to_database(conn, image_name, detection_data):
    cursor = conn.cursor()
    for detection in detection_data:
        detection['bbox'] = list(map(int, detection['bbox']))
    detections_json = json.dumps(detection_data)
    cursor.execute('''
        INSERT INTO model_detection (image_name, detections)
        VALUES (?, ?)
    ''', (image_name, detections_json))
    conn.commit()


def video_reader(cap, frame_queue):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)
    frame_queue.put(None)


def process_frames(frame_queue, conn, read_queue):
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        detected_objects = detect_faces(frame)
        print(detected_objects)
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        image_name = f"image_{str(int(frame_number)).zfill(3)}.jpg"
        save_to_database(conn, image_name, detected_objects)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        read_queue.put(image_name)


def database_reader(conn, read_queue):
    cursor = conn.cursor()
    while True:
        image_name = read_queue.get()
        if image_name is None:
            break
        cursor.execute('''
            UPDATE model_detection
            SET is_read = 1
            WHERE image_name = ?
        ''', (image_name,))
        conn.commit()



conn = sqlite3.connect('detection_database.db', check_same_thread=False)
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS model_detection (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_name TEXT NOT NULL,
        detections TEXT,
        is_read INTEGER DEFAULT 0
    )
''')
conn.commit()

frame_queue = Queue()
read_queue = Queue()

cap = cv2.VideoCapture("video.mp4")

video_reader_thread = threading.Thread(target=video_reader, args=(cap, frame_queue))
process_frames_thread = threading.Thread(target=process_frames, args=(frame_queue, conn, read_queue))
database_reader_thread = threading.Thread(target=database_reader, args=(conn, read_queue))

video_reader_thread.start()
process_frames_thread.start()
database_reader_thread.start()

video_reader_thread.join()
frame_queue.put(None)
process_frames_thread.join()
read_queue.put(None)
database_reader_thread.join()

conn.close()
cap.release()
