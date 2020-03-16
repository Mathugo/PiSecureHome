import face_recognition
import cv2
import numpy as np
import os
import glob
from imutils.video import VideoStream
import time

class ImgRecognition:

    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

    def gatherData(self, path_dataset):  
        self.path_dataset = path_dataset  
        self.dirname = os.path.dirname(__file__)
        self.path = os.path.join(self.dirname, self.path_dataset)

        self.list_of_files = [f for f in glob.glob(self.path+'*.jpg')]
        self.names = self.list_of_files.copy()
        self.number_files = len(self.list_of_files)
        self.loadRecognition()

    def loadRecognition(self):    
        print("[*] "+str(self.number_files)+" file to load")
        for i in range(self.number_files):
            print("[!] Loading recognition file from "+str(self.list_of_files[i]))
            globals()['image_{}'.format(i)] = face_recognition.load_image_file(self.list_of_files[i])
            globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0]
            self.known_face_encodings.append(globals()['image_encoding_{}'.format(i)])
            self.names[i] = self.names[i].replace(self.path_dataset,'')
            self.known_face_names.append(self.names[i])
        print("[*] Done loading dataset")    

    def getKnownFaceEncoding(self):
        return self.known_face_encodings    
    def getKnownFaceNames(self):
        return self.known_face_names        

class Image:
    def __init__(self,frame, reco):
        self.recognition = reco
        self.known_face_encodings = self.recognition.getKnownFaceEncoding()
        self.known_face_names = self.recognition.getKnownFaceNames()
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
            self.face_names.append(name)

    def display(self):
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

            name = name.replace(".\dataset\\", "")
            name = name.replace(".\dataset/","")
            name = name.replace(".jpg",'')
            name = name.replace(".png",'')
            n = name.split('\\')
            name = n[0]
            cv2.putText(self.img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            cv2.imshow('Video', self.img)


def main():
    print("[*] Starting webcam ..")
    video_capture = cv2.VideoCapture(0)
    print("[*] Warming-up ..")
    time.sleep(2)
    path_hugo = "dataset/hugo/"
    #path_alex = "dataset/alex/"
    reco = ImgRecognition()
    reco.gatherData(path_hugo)
    #reco.gatherData(path_alex)
    
    process_this_frame = True
    ret, frame = video_capture.read()
    cv2.imshow('Video', frame)

    print("[*] Waiting recognition ..")
    while True:
        ret, frame = video_capture.read()
        img = Image(frame, reco)
        if process_this_frame:
            img.processRecognition()

        process_this_frame = not process_this_frame
        img.display()
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        
    video_capture.release()
    cv2.destroyAllWindows()    

main()    
