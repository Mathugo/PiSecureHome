import numpy as np
import cv2
import face_recognition

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
            name = self.parseName(name)
            cv2.putText(self.img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
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
