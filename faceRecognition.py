import numpy as np
import os
import face_recognition
import glob

class FaceRecognition:

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
            f = face_recognition.face_encodings(globals()['image_{}'.format(i)])
            if len(f) == 0:
                print("[!] Face not detected on "+str(self.list_of_files[i]))
            else:
                globals()['image_encoding_{}'.format(i)] = f[0]
                self.known_face_encodings.append(globals()['image_encoding_{}'.format(i)])
                #self.names[i] = self.names[i].replace(self.path_dataset,'')
                self.known_face_names.append(self.names[i])
                
        print("[*] Done loading dataset")    

    def getKnownFaceEncoding(self):
        return self.known_face_encodings    
    def getKnownFaceNames(self):
        return self.known_face_names        
