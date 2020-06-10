import numpy as np
import time
import cv2
import imutils
from threading import Thread

class otherDetection(Thread):
    def __init__(self, prototxt, model, frame):
        Thread.__init__(self)
        self.frame = frame
        self.limitConfidence = 0.3
        self.objects = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
        self.colors = np.random.uniform(0, 255, size=(len(self.objects), 3))
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
        self.frame = imutils.resize(frame, width=400)
        
    def run(self):
        (self.h, self.w) = self.frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(self.frame, (300, 300)),
        0.007843, (300, 300), 127.5)
        self.net.setInput(blob)

        self.detections = self.net.forward()
        self.draw_rect()
        
    def draw_rect(self):
        for i in np.arange(0, self.detections.shape[2]):
            confidence = self.detections[0, 0, i, 2]
            if confidence > self.limitConfidence:

                idx = int(self.detections[0, 0, i, 1])  
                box = self.detections[0, 0, i, 3:7] * np.array([self.w, self.h, self.w, self.h])
                (startX, startY, endX, endY) = box.astype("int")
                label = "{}: {:.2f}%".format(self.objects[idx],
				confidence * 100)
                cv2.rectangle(self.frame, (startX, startY), (endX, endY),
				self.colors[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(self.frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[idx], 2)
        cv2.imshow('Video',self.frame)