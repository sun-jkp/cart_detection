import cv2
import numpy as np
import json

# Load image
img = cv2.imread('p1.jpg')  # Change this to your image path
img = cv2.resize(img, (640, 480))
img_copy = img.copy()

# Variables
current_points = []
all_polygons = []  # Store all parking slot polygons

def draw_polygon(event, x, y, flags, param):
    global current_points, img_copy, all_polygons

    if event == cv2.EVENT_LBUTTONDOWN:  # Left click: add point
        current_points.append((x, y))
        cv2.circle(img_copy, (x, y), 3, (0, 0, 255), -1)

    elif event == cv2.EVENT_RBUTTONDOWN:  # Right click: finish current polygon
        if len(current_points) >= 3:  # At least 3 points to form a polygon
            # Draw polygon
            cv2.polylines(img, [np.array(current_points)], isClosed=True, color=(0, 255, 0), thickness=2)
            # Save polygon points
            all_polygons.append(current_points.copy())
            # Reset for next polygon
            current_points = []
            img_copy = img.copy()

cv2.namedWindow('Parking Slots')
cv2.setMouseCallback('Parking Slots', draw_polygon)

while True:
    cv2.imshow('Parking Slots', img_copy)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # Press 'q' to quit and save
        break

# Save polygons to JSON
parking_slots = {'slots': []}
for idx, polygon in enumerate(all_polygons):
    slot = {'id': idx, 'points': polygon}
    parking_slots['slots'].append(slot)

with open('parking_slots.json', 'w') as f:
    json.dump(parking_slots, f, indent=4)

print("Saved polygons to parking_slots.json")

cv2.destroyAllWindows()
