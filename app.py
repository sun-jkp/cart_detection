import parking_detection
import cv2
import requests
import numpy as np
import json
# import time
# Setup
image_path = 'p1.jpg'
# model_path = 'best.pt'
model_path = 'best_ncnn_model'
parking_station_1_path = 'parking_slots.json'
# parking_station_2_path = 'parking_slots.json'
# parking_station_3_path = 'parking_slots.json'
class_names = ['cart_empty', 'cart_loaded', 'sign_cart', 'cart_unknow']  # Adjust if different


#parking station 1
with open(parking_station_1_path, 'r') as f:
    parking_slot_station_1 = json.load(f)
# parking station 2
# with open(parking_station_2_path, 'r') as f:
#     parking_slot_station_2 = json.load(f)
# # parking station 3
# with open(parking_station_3_path, 'r') as f:
#     parking_slot_station_3 = json.load(f)

# Create detector
# detector = parking_detection.ParkingStationDetector(model_path, json_path, class_names)
detector = parking_detection.ParkingStationDetector(model_path, class_names)
while True:
    # Get image from camera
    result = requests.get('http://10.14.73.253/liveimage.jpg')
    if(result.status_code == 200):
        img_array = np.frombuffer(result.content, np.uint8)
        input_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        input_img_copy = input_img.copy()
        # cv2.imshow("Live from SR-2000W", input_img)
        # Run detection
        detector.run(input_img_copy, parking_station_1_path)
        img = detector.get_result()
        cv2.imshow('Parking Station Detection', img)
        # time.sleep(1)
        
    key = cv2.waitKey(1000) & 0xFF

    if key == ord('q'):  # Press 'q' to quit and save
        break
    # cv2.waitKey(0)
cv2.destroyAllWindows()