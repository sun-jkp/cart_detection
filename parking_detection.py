import cv2
import json
import numpy as np
import shapely.geometry
from ultralytics import YOLO

class ParkingStationDetector:
    # def __init__(self, image_path, model_path, json_path, class_names):
    # def __init__(self, model_path, json_path, class_names):
    def __init__(self, model_path, class_names):
        # Load image
        # self.image = cv2.imread(image_path)
        # self.image = cv2.resize(self.image, (640, 480))
        # self.img_result = self.image.copy()
        
        # Load YOLO model
        self.model = YOLO(model_path)
        
        # Class names for detection
        self.class_names = class_names
        
        
        # self.slot_polygons = []
        # self.load_parking_slots()
            
        # IoU threshold for matching cart to slot
        self.iou_threshold = 0.1

        # Detection results (after predict)
        self.detections = []
        
    def load_parking_slots(self, parking_slots):
        if isinstance(parking_slots, str):
            # Load parking slots from JSON
            with open(parking_slots, 'r') as f:
                self.parking_slots = json.load(f)
        else:
            self.parking_slots = parking_slots
            
        self.slot_polygons = []
        for slot in self.parking_slots['slots']:
            points = slot['points']
            polygon = shapely.geometry.Polygon(points)
            self.slot_polygons.append({
                'id': slot['id'],
                'points': points,
                'polygon': polygon,
                'status': 'empty'
            })
        
    def detect_carts(self, img):
        # self.load_parking_slots()
        self.image = None
        self.detections = []
        if(img is str):
            self.image = cv2.imread(img)
        else:
            self.image = img
            
        self.image = cv2.resize(self.image, (640, 480))
        self.img_result = self.image.copy()
        results = self.model.predict(self.image, verbose=False)

        for box in results[0].boxes:  # Assume first image
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = int(box.cls[0])
            if conf >= 0.5:  # Confidence threshold
                if cls != 2:
                    if conf < 0.7:
                        cls = 3
                self.detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'conf': float(conf),
                    'cls': cls
                })
            
    def match_carts_to_slots(self):
        for detection in self.detections:
            x1, y1, x2, y2 = detection['bbox']
            box_polygon = shapely.geometry.box(x1, y1, x2, y2)

            for slot in self.slot_polygons:
                slot_polygon = slot['polygon']

                if box_polygon.is_valid and slot_polygon.is_valid:
                    intersection_area = box_polygon.intersection(slot_polygon).area
                    union_area = box_polygon.union(slot_polygon).area
                    iou = intersection_area / union_area if union_area != 0 else 0

                    if iou > self.iou_threshold:
                        slot['status'] = 'not_empty'
                        
    def draw_results(self):
        # Draw parking slots
        for slot in self.slot_polygons:
            points = np.array(slot['points'], np.int32).reshape((-1, 1, 2))

            if slot['status'] == 'not_empty':
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            # Draw polygon
            cv2.polylines(self.img_result, [points], isClosed=True, color=color, thickness=1)

            # Put label
            text = f"Slot {slot['id']}: {slot['status']}"
            centroid_x = int(np.mean([p[0] for p in slot['points']]))
            centroid_y = int(np.mean([p[1] for p in slot['points']]))
            cv2.putText(self.img_result, text, (centroid_x - 40, centroid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw YOLO detection boxes
        for detection in self.detections:
            x1, y1, x2, y2 = detection['bbox']
            cls = detection['cls']
            label = self.class_names[cls]
            # color = (0, 0, 255)  # Red for detected cart
            if(label == 'cart_empty'):
                color = (180, 235, 176)
            elif(label == 'cart_loaded'):
                color = (255, 246, 212)
            elif(label == 'sign_cart'):
                color = (129, 238, 236)
            else:
                color = (241, 135, 143)
            cv2.rectangle(self.img_result, (x1, y1), (x2, y2), color, 1)
            cv2.putText(self.img_result, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
    def show_result(self):
        cv2.imshow('Parking Station Detection', self.img_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run(self, img, parking_slots, show=False):
        self.load_parking_slots(parking_slots)
        self.detect_carts(img)
        self.match_carts_to_slots()
        self.draw_results()
        if(show):
            self.show_result()
        
    def get_result(self):
        return self.img_result