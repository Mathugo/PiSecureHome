import cv2
import time
from Face import *
import argparse
import pickle
import imutils
from imutils.video import VideoStream, FPS
import time
from load_model import *

class Application:
    def __init__(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-d", "--display", type=str, default="yes",
            help="display image `yes` or `no`")
        ap.add_argument("-c", "--cascade", type=str, default="hog", required=True,
            help="cascade path for opencv")
        ap.add_argument("-m", "--model", type=str, default="model.tflite", required=True,
            help="lite model path for the face detection")
        self.args = vars(ap.parse_args()) 
        self.exit = False
        self.face_detector = Face(self.args["cascade"])
        self.lite_model = Lite_Model(self.args["model"])

        self.init_webcam()
        self.fps = FPS().start()
    
    def init_webcam(self):
        print("[*] Starting webcam ..")
        print("[*] Warming-up ..")
        self.video_capture = VideoStream(src=0).start()
        self.writer= None
        time.sleep(2)
    
    def getFrame(self):
        self.frame = self.video_capture.read()
        #self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.frame = imutils.resize(self.frame, width=700)
        #self.rescale = self.frame.shape[1] / float(self.frame.shape[1])
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

            if self.face_detector.detectFace():
                face = self.face_detector.extract_faces()
                self.lite_model.set_image(face)
                results = self.lite_model.classify_image()
                elapsed_ms = (time.time() - start_time) * 1000
                label_id, prob = results[0]
                print("[*] Label id {} Proba {} Elapsed time {}".format(label_id, prob, elapsed_ms))
                if self.args["display"] == "yes":
                    if label_id:
                        self.face_detector.draw_faces_name("Hugo "+str(prob*100)+"%")
                    else:
                        self.face_detector.draw_faces_name("Unknown "+str(prob*100)+"%")

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