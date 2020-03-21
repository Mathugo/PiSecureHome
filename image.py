import numpy as np
import cv2
import face_recognition
from speech import *
from datetime import datetime
import time
from otherDetection import otherDetection
import imutils

class Image:
    
    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.delaySpeaking = 10
        self.delayNames = {}
        self.path_cascade = "dataset/cascade/haarcascade_frontalface_default.xml"
        if self.args["detection_method"] == "cascade":
            self.face_cascade = cv2.CascadeClassifier(self.path_cascade)
        print("[*] Using {} method".format(self.args['detection_method']))

    def setFrame(self, frame, rescale):
        self.frame = frame
        self.rescale = rescale
    
    def detectFace(self):
        print("[INFO] recognizing faces...")

        if self.args["detection_method"] == "cascade":
            self.cascade()
        else:
            self.boxes = face_recognition.face_locations(self.frame,
	        model=self.args["detection_method"])

        self.encodings = face_recognition.face_encodings(self.frame, self.boxes)
        print("[*] Done")
        # initialize the list of names for each face detected
    def cascade(self):
        rects = self.face_cascade.detectMultiScale(
                self.frame,
                scaleFactor=1.3,
                minNeighbors=1,
                minSize=(30, 30))
        self.boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

    def checkNames(self):
        self.names = []
        for encoding in self.encodings:
	    # attempt to match each face in the input image to our known
	    # encodings
            matches = face_recognition.compare_faces(self.data["encodings"],
            encoding)
            name = "Unknown"
            speech_text = "Une personne inconnu a été détecté"
            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = self.data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                # determine the recognized face with the largest number of
                # votes (note: in the event of an unlikely tie Python will
                # select first entry in the dictionary)
                name = max(counts, key=counts.get)
                speech_text = name+" est devant votre porte"
                print("[!] Name detected : "+name)

            self.speak(speech_text, name)
            # update the list of names
                
            self.names.append(name)
            if self.args["display"] == "yes":
                self.draw_boxes(name)
                self.display()

    def speak(self, text_speech, name):
        if name not in self.delayNames or (time.time() - self.delayNames.get(name)) > self.delaySpeaking:
            s = Speech(text_speech, language_code='fr')
            s.start()
            self.delayNames[name] = time.time()
    def draw_boxes(self, name):
        for ((top, right, bottom, left), name) in zip(self.boxes, self.names):
	    # draw the predicted face name on the image
            top = int(top * self.rescale)
            right = int(right * self.rescale)
            bottom = int(bottom * self.rescale)
            left = int(left * self.rescale)

            cv2.rectangle(self.frame, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(self.frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (0, 255, 0), 2)
            # show the output image

    def display(self):
        cv2.imshow("Webcam", self.frame)
