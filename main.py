import cv2
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
def detect_fire_and_estimate_distance(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    lower_bound = np.array([0, 100, 150], dtype=np.uint8)
    upper_bound = np.array([35, 255, 255], dtype=np.uint8)
    _, lum_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    color_mask = cv2.inRange(hsv, lower_bound, upper_bound)
    combined_mask = cv2.bitwise_and(color_mask, lum_mask)

    kernel = np.ones((7, 7), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            x, y, w, h = cv2.boundingRect(contour)
            distance = 1000 / area
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, f"Fire Detected - {distance:.2f} meters", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    return frame, combined_mask


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    detected_frame, fire_mask = detect_fire_and_estimate_distance(frame)
    cv2.imshow('Fire Detection', detected_frame)
    cv2.imshow('Fire Mask', fire_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()