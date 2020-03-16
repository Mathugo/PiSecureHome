import cv2
import time
from imageRecognition import *
from image import *

class Application:

    def __init__(self):
        print("[*] Starting webcam ..")
        print("[*] Warming-up ..")
        self.exit = False
        self.video_capture = cv2.VideoCapture(0)
        time.sleep(2)
        self.recognition = ImgRecognition()

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
        process_this_frame = True
        print("[*] Waiting recognition ..")
        print("[!] Press q to abort")
        while self.exit == False:
            self.getFrame()
            self.img = Image(self.frame, self.recognition)
            if process_this_frame:
                self.img.processRecognition()
            process_this_frame = not process_this_frame
            self.img.display()
            self.keyWait()
            
    def exitSafely(self):
        self.video_capture.release()
        cv2.destroyAllWindows()      
    
