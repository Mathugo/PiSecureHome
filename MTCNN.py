from mtcnn import MTCNN
from matplotlib import pyplot
from matplotlib.patches import Rectangle, Circle


class Image:
    def __init__(self, filename):
        self.img = pyplot.imread(filename)
        self.faces = 0
        self.faces_rect = []

    def detectFaces(self):
        detector = MTCNN(min_face_size=50)# Minimum box size for detecting face -> increase performance
        self.faces = detector.detect_faces(self.img)
        for face in self.faces:
            print("Face detected : "+str(face))

    def draw_image_with_boxes(self):
        pyplot.imshow(self.img)
        ax = pyplot.gca() #get the context for drawing boxes
        for result in self.faces:
            print("printing boxes ..")
            x, y , width, height = result['box'] # Get coordinates
            rect = Rectangle((x,y), width, height, fill=False, color='red') # create shape rectangle
            ax.add_patch(rect) #Draw the box
            for key, value in result['keypoints'].items():
                dot = Circle(value, radius=2, color='red')
                ax.add_patch(dot)

        pyplot.show()    
    def draw_faces(self):
        for i in range(len(self.faces)):
            x1, y1, width, height = self.faces[i]['box']
            x2, y2 = x1 + width, y1 + height
            # define subplot
            pyplot.subplot(1, len(self.faces), i+1)
            pyplot.axis('off')
            # plot face
            self.faces_rect.append(self.img[y1:y2, y1:y2])
            pyplot.imshow(self.faces_rect[i])
        pyplot.show()    

          

def main():
    filename = "data/hugo/hugo1.jpg"
    image1 = Image(filename)
    image1.detectFaces()
    #image1.draw_image_with_boxes()
    image1.draw_faces()
main()        