import cv2
import time
from faceRecognition import *
from image import *
from otherDetection import *

path_prototxt = "dataset/object_detection/MobileNetSSD_deploy.prototxt.txt"
path_model = "dataset/object_detection/MobileNetSSD_deploy.caffemodel"

class Application:

    def __init__(self):
        print("[*] Starting webcam ..")
        print("[*] Warming-up ..")
        self.exit = False
        self.video_capture = cv2.VideoCapture(0)
        time.sleep(2)
        self.recognition = FaceRecognition()

    def load(self, path_dataset):
        self.recognition.gatherData(path_dataset)

    def getFrame(self):
        ret, self.frame = self.video_capture.read()

    def keyWait(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.exit= True
            
    def run(self):
        self.getFrame()
        cv2.imshow('Video', self.frame)
        self.img = Image(self.recognition)
        process_this_frame = True
        print("[*] Waiting recognition ..")
        print("[!] Press q to abort")

        while self.exit == False:
            self.getFrame()
            self.img.loadFrame(self.frame)

            if process_this_frame:
                self.img.processRecognition()
                self.img.detectOther()
                self.img.display()

            process_this_frame = not process_this_frame

            self.keyWait()

            
    def exitSafely(self):
        print("[!] Exiting ..")
        self.video_capture.release()
        cv2.destroyAllWindows()
        print("[*] Done")      
    
