import cv2

class Face:
    def __init__(self, path_cascade):
        print("[*] Loading cascade ..")
        self.faceCascade = cv2.CascadeClassifier(path_cascade)
        #self.image_size = (600, 400)
        self.image_to_detect_size = (224, 224)
        print("[*] Done")

    def load_image(self, imagePath):
        self.image = cv2.imread(imagePath)
        #self.image = cv2.resize(self.image, self.image_size)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def setImage(self, image):
        self.image = image
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def getImage(self):
        return self.image

    def detectFace(self):
        self.faces = self.faceCascade.detectMultiScale(
        self.gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
        )
        print("Found {0} Faces!".format(len(self.faces)))

        if (len(self.faces)) >=1:
            return len(self.faces)
        else:
            return False

    
    def draw_faces_name(self, label):
        for (x, y, w, h) in self.faces:
            cv2.rectangle(self.image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(self.image, label, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        #status = cv2.imwrite('E:\ESIREM\Python\PiSecureHome\pi4\faces_detected.jpg', self.image)
        #print ("Image faces_detected.jpg written to filesystem: ",status)

    def extract_faces(self):
        for (x, y, w, h) in self.faces:
            
            #cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_color = self.image[y:y + h, x:x + w]
            face_color = cv2.resize(face_color, self.image_to_detect_size)
            #face_color = cv2.resize(face_color, self.image_to_detect_size) 
            #print("[INFO] Object found. Saving locally.") 
            return face_color
            #cv2.imwrite(str(w) + str(h) + '_faces.jpg', face_color) 

