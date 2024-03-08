import cv2
from threading import Thread
from queue import Queue

class FrameCapture:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.q = Queue()
        self.running = True

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
            if not self.q.full():
                self.q.put(frame)

    def read(self):
        return self.q.get()

    def stop(self):
        self.running = False
        self.cap.release()