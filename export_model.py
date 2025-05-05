from ultralytics import YOLO
import requests
import cv2
import numpy as np
# path = 'best_ncnn_model'
# model = YOLO(path, task='detect')
# model.export(format='ncnn')

result = requests.get('http://10.14.73.253/liveimage.jpg')
img_array = np.frombuffer(result.content, np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

cv2.imshow("Live from SR-2000W", img)
cv2.waitKey(0)
cv2.destroyAllWindows()