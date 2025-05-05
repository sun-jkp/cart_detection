import cv2
import json
import numpy as np
from ultralytics import YOLO 
import shapely.geometry

img_path = 'p1.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, (640, 480))

# Load parking slots from JSON
with open('parking_slots.json', 'r') as f:
    parking_slots = json.load(f)
    

model = YOLO('best.pt')  # Load the YOLOv8 model
results = model(img, show=False, conf=0.3)  # Perform inference
# print(len(results))
# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputsw
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    name_cls = result.names  # Class names
    # print(name_cls)  # Print class names
    # result.show()  # display to screen
    # result.save(filename="result.jpg")  # save to disk

for box in results[0].boxes:

    # Print box coordinates and confidence
    # print(f"Box: {box.xyxy[0]}, Confidence: {box.conf[0]}")
    # Draw bounding boxes on the image
    # 0: Empty
    # 1: loaded
    # 2: sign
    if box.cls[0] == 0:
        cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])), (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0, 255, 0), 2)
    if box.cls[0] == 1:
        cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])), (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0, 0, 255), 2)
    if box.cls[0] == 2:
        cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])), (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)

    # Rectangle box
    box = shapely.geometry.box(int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3]))

    for slot in parking_slots['slots']:
        points = np.array(slot['points'], np.int32)
        # points = points.reshape((-1, 1, 2))
        # Irregular parking slot
        polygon = shapely.geometry.Polygon(points)
        intersection = box.intersection(polygon).area
        union = box.union(polygon).area
        iou = intersection / union
        print(iou)
    print('-------------')

# print(results.xywh)



# Draw each parking slot polygon
for slot in parking_slots['slots']:
    points = np.array(slot['points'], np.int32)
    points = points.reshape((-1, 1, 2))
    # print(points)
    # print('------------------')
    cv2.polylines(img, [points], isClosed=True, color=(7, 202, 255), thickness=2)
        
cv2.imshow('Image with Coordinates', img)

cv2.waitKey(0)
cv2.destroyAllWindows()