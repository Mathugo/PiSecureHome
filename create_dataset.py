from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
	help = "path to where the face cascade resides")
ap.add_argument("-o", "--output", required=True,
	help="path to output directory")
args = vars(ap.parse_args())

# load OpenCV's Haar cascade for face detection from disk
detector = cv2.CascadeClassifier(args["cascade"])
# initialize the video stream, allow the camera sensor to warm up,
# and initialize the total number of example faces written to disk
# thus far
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0) # Warm up
total = 0

class Image:
  def __init__(self, frame):
    self.orig = frame.copy()
    self.img = imutils.resize(frame, width=400)
    self.scale = 1.1
    self.minN = 5
    self.minSize = (30,30)

  def detectFace(self):
    print("Detecting face ..")
    rects = detector.detectMultiScale(
    cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), scaleFactor=self.scale, 
    minNeighbors=self.minN, minSize=self.minSize)
    print("Done, now drawing rects ..")
    for (x, y, w, h) in rects:
      cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    print("Done")  

  def showImg(self):
    cv2.imshow("Frame", self.img)
  
  def writeImg(self):
    global total
    p = os.path.sep.join([args["output"], "{}.png".format(
    str(total).zfill(5))])
    cv2.imwrite(p, self.orig)
    total += 1
    
def create_dataset(self):    
  while True:
    frame = vs.read()
    img = Image(frame)
    #img.detectFace()
    img.showImg()
    key = cv2.waitKey(1) & 0xFF

    if key == ord("k"):
        img.writeImg()
    elif key == ord("q"):
        break

    print("[INFO] {} face images stored".format(total))
    print("[INFO] cleaning up...")

  #cv2.destroyAllWindows()
  #vs.stop()

