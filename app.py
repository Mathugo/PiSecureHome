import cv2
import time
from image import *
from otherDetection import *
import argparse
import pickle
import imutils
from imutils.video import VideoStream, FPS
import time

class Application:

    def __init__(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-e", "--encodings", required=True,
            help="path to serialized db of facial encodings")
        ap.add_argument("-d", "--detection-method", type=str, default="hog",
            help="face detection model to use: either `hog` or `cnn` or `cascade`")
        ap.add_argument("-p", "--display", type=str, default="yes",
            help="display image `yes` or `no`")
        self.args = vars(ap.parse_args())
        self.exit = False
        self.init_webcam()
        self.fps = FPS().start()
        print("[*] Loading encodings ...")
        self.data = pickle.loads(open(self.args["encodings"], "rb").read())
        print("[*] Done")
        self.image_reco = Image(self.args, self.data)

    def init_webcam(self):
        print("[*] Starting webcam ..")
        print("[*] Warming-up ..")
        self.video_capture = VideoStream(src=0).start()
        self.writer= None
        time.sleep(2)


    def getFrame(self):
        self.frame = self.video_capture.read()
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.frame = imutils.resize(self.frame, width=750)
        self.rescale = self.frame.shape[1] / float(self.frame.shape[1])
        self.image_reco.setFrame(self.frame, self.rescale)

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
            self.image_reco.detectFace()
            self.image_reco.checkNames()
            self.image_reco.display()
            self.fps.update()
            self.keyWait()
            self.fps.stop()
            print("[INFO] elasped time: {:.2f}".format(self.fps.elapsed()))
            print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))

            
    def exitSafely(self):
        print("[!] Exiting ..")
        self.video_capture.stop()
        cv2.destroyAllWindows()
        print("[*] Done")      
    
