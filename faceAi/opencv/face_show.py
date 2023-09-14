import face_recognition
import cv2 as cv

def drawFaceFunc():
    image = face_recognition.load_image_file("./my_img/03.jpg")
    image = face_recognition.load_image_file("./face3.jpg")
    image = face_recognition.load_image_file("./test/03.jpg")
    face_loactions = face_recognition.face_locations(image)
    for one in face_loactions:
        y0, x1 ,y1 ,x0 = one
        print(y0, x1 ,y1 ,x0)
        cv.rectangle(image,pt1=(x0,y0),pt2=(x1,y1),color=(0,0,255),thickness=3)
        cv.putText(image,'face', (x0 + 5, y0 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv.imshow("bbb",image)
    if cv.waitKey(0) & 0xFF ==("q"):
        cv.destroyAllWindows()    


if __name__ == "__main__":
   drawFaceFunc()