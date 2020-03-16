import numpy as np
import cv2
import face_recognition
from speech import *
from datetime import datetime
import time
from otherDetection import otherDetection
import imutils

path_prototxt = "dataset/object_detection/MobileNetSSD_deploy.prototxt.txt"
path_model = "dataset/object_detection/MobileNetSSD_deploy.caffemodel"
class Image:
    
    def __init__(self, reco):
        self.recognition = reco
        self.delaySpeaking = 15 #Delay between each same speaking
        self.known_names = []
        self.delayNames = {}


        self.known_face_encodings = self.recognition.getKnownFaceEncoding()
        self.known_face_names = self.recognition.getKnownFaceNames()

        self.initKnownNames()
        self.initDelayFromNames()

    def initKnownNames(self):
        for path_name in self.known_face_names:
            name = self.parseName(path_name)
            if name not in self.known_names:
                print(name+" added")
                self.known_names.append(name)

    def initDelayFromNames(self):
        for i in range(0, len(self.known_names)-1):
            self.delayNames[self.known_names[i]] = 0
        print("Adding unknown delay ..")
        self.delayNames['Unknown'] = 0    

    def loadFrame(self, frame):
        self.img = frame
        self.small_frame = cv2.resize(self.img, (0,0), fx=0.25, fy=0.25) # 1/4 from orig size for better perfs
        self.rgb_small_frame = self.small_frame[:, :, ::-1] # Convert img from BGR(opencv) to RGB color (face reco)
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []

    def processRecognition(self):
        #Find if the face is a match for the known face(s)
        self.face_locations = face_recognition.face_locations(self.rgb_small_frame)
        self.face_encodings = face_recognition.face_encodings(self.rgb_small_frame, self.face_locations)
        for face_encoding in self.face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            text_speech = "Une personne inconnu a été détectée"
            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                print("[*] Img detected : "+name)
                self.parsedName = self.parseName(name)
                print("[!] Personn : "+self.parsedName)
                text_speech = self.parsedName+" est devant votre porte"

            self.speak(name, text_speech)
           
            self.face_names.append(name)
        self.draw_names()

    def speak(self, name, text_speech):
        if name != "Unknown":                    
            if (time.time() - self.delayNames[self.parsedName]) > self.delaySpeaking or self.delayNames[self.parsedName] == 0:
                s = Speech(text_speech, language_code='fr')
                s.start()
                self.delayNames[self.parsedName] = time.time()
        else: #name eq unkown
            if (time.time() - self.delayNames["Unknown"]) > self.delaySpeaking or self.delayNames["Unknown"] == 0:
                s = Speech(text_speech, language_code='fr')
                s.start()
                self.delayNames["Unknown"] = time.time()

    def draw_names(self):
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4    
            # Draw a box around the face
            cv2.rectangle(self.img, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(self.img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            if self.face_names != "Unknown":
                cv2.putText(self.img, self.parsedName, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            else:
                cv2.putText(self.img, "Unkown", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    def display(self):
        cv2.imshow('Video', self.img)

    def parseName(self, name):
        name = name.replace(".\dataset\\", "")
        name = name.replace(".\dataset/","")
        name = name.replace(".jpg",'')
        name = name.replace(".png",'')
        n = name.split('\\')
        name = n[len(n)-2]
        name = name.split('/')
        name = name[1]
        return name

    def detectOther(self):
        self.limitConfidence = 0.3
        self.objects = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
        self.colors = np.random.uniform(0, 255, size=(len(self.objects), 3))
        self.net = cv2.dnn.readNetFromCaffe(path_prototxt, path_model)

        (self.h, self.w) = self.img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(self.img, (300, 300)),
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
                cv2.rectangle(self.img, (startX, startY), (endX, endY),
				self.colors[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(self.img, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[idx], 2)

    def getFrame(self):
        return self.img