from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

class Encoding:
    
    def __init__(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--dataset", required=True,
        help="path to input directory of faces + images")
        ap.add_argument("-e", "--encodings", required=True,
        help="path to serialized db of facial encodings")
        ap.add_argument("-d", "--detection-method", type=str, default="cnn",
        help="face detection model to use: either `hog` or `cnn`")
        self.args = vars(ap.parse_args())

    def processing_img(self):
        print("[INFO] quantifying faces...")
        self.imagePaths = list(paths.list_images(self.args["dataset"]))
        # initialize the list of known encodings and known names
        self.knownEncodings = []
        self.knownNames = []

        # loop over the image paths
        for (i, self.imagePath) in enumerate(self.imagePaths):
        # extract the person name from the image path
            print("[INFO] processing image {}/{}".format(i + 1,
            len(self.imagePaths)))
            name = self.imagePath.split(os.path.sep)[-2]
        # load the input image and convert it from BGR (OpenCV ordering)
        # to dlib ordering (RGB)
            image = cv2.imread(self.imagePath)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            boxes = face_recognition.face_locations(rgb,
                model=self.args["detection_method"])
            # compute the facial embedding for the face
            encodings = face_recognition.face_encodings(rgb, boxes)
            # loop over the encodings
            for encoding in encodings:
                # add each encoding + name to our set of known names and
                # encodings
                self.knownEncodings.append(encoding)
                self.knownNames.append(name)

    def serializing_img(self):
        print("[INFO] serializing encodings...")
        data = {"encodings": self.knownEncodings, "names": self.knownNames}
        f = open(self.args["encodings"], "wb")
        f.write(pickle.dumps(data))
        f.close()
        print("[*] Done")

    def run(self):
        self.processing_img()
        self.serializing_img()
  
   

