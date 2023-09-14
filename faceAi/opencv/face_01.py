import cv2 as cv
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
#img = ins_get_image('t2')
img = cv.imread('./test/03.jpg')
faces = app.get(img)
rimg = app.draw_on(img, faces)
cv.imshow('aaa',rimg)
#cv2.imwrite("./t3_output.jpg", rimg)