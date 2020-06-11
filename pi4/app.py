import cv2
import time
import argparse
import numpy
import picamera
import io
from PIL import Image

import imutils
from imutils.video import VideoStream, FPS
import time
from LiteModel import *
from load_model_2 import *
from Face import *
from tflite_runtime.interpreter import Interpreter

WIDTH= 224
HEIGHT= 224

class Application:
    def __init__(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-d", "--display", type=str, default="no",
            help="display image `yes` or `no`")
        ap.add_argument("-c", "--cascade", type=str, default="hog", required=True,
            help="cascade path for opencv")
        ap.add_argument("-m", "--model", type=str, default="model.tflite", required=True,
            help="lite model path for the face detection")
        self.args = vars(ap.parse_args()) 
        self.exit = False
        self.face_detector = Face(self.args["cascade"])
        self.init_model()
        self.init_webcam()
        self.fps = FPS().start()
    
    def init_model(self):
        print("[!] Loading {} ..".format(str(self.args["model"])))
        self.interpreter = Interpreter(self.args["model"])
        print("[*] Done")
        self.interpreter.allocate_tensors()
        _, height, width, _ = self.interpreter.get_input_details()[0]['shape']
    def init_webcam(self):
        print("[*] Starting webcam ..")
        print("[*] Warming-up ..")
        self.video_capture = VideoStream(src=0).start()
        self.writer= None
        time.sleep(2)
    
    def getFrame(self):
        self.frame = self.video_capture.read()
        self.frame = imutils.resize(self.frame, width=700)
    def setFrame(self, frame):
        self.frame = frame
    def keyWait(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.exit= True

    def run(self):
        self.getFrame()
        process_this_frame = True
        print("[*] Waiting recognition ..")
        print("[!] Press q to abort")

        while self.exit == False:
            self.getFrame()
            self.face_detector.setImage(self.frame)
            start_time = time.time()

            if self.face_detector.detectFace() >= 1:
                face = self.face_detector.extract_faces()
                label, prob = run_model(self.interpreter, self.args["model"], face)

                if self.args["display"] == "yes":
                    self.face_detector.draw_faces_name(label+" "+str(prob*100)+"%")

            self.fps.update()
            self.keyWait()
            self.fps.stop()
            if self.args["display"] == "yes":
                self.setFrame(self.face_detector.getImage())
                self.display()
                
            print("[INFO] elasped time: {:.2f}".format(self.fps.elapsed()))
            print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))

    def exitSafely(self):
        print("[!] Exiting ..")
        self.video_capture.stop()
        cv2.destroyAllWindows()
        print("[*] Done")    

    def display(self):
        cv2.imshow("Webcam", self.frame)

ap = Application()
ap.run()