import cv2
import numpy as np

LEMON_YELLOW_BGR = [0, 247, 255]
DELTA_H = 10
S_MIN = 120
V_MIN = 140
MIN_AREA = 2000
EXCLUDE_SKIN = True

def get_yellow_limits(color_bgr, delta=10, s_min=120, v_min=140):
    c = np.array(color_bgr, dtype=np.uint8).reshape(1, 1, 3)
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    h = int(hsv[0, 0, 0])
    h_lo = max(h - delta, 0)
    h_hi = min(h + delta, 179)
    lower = np.array([h_lo, s_min, v_min], dtype=np.uint8)
    upper = np.array([h_hi, 255, 255], dtype=np.uint8)
    return lower, upper

def skin_mask_ycrcb(frame_bgr):
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    return cv2.inRange(ycrcb, lower, upper)

def clean_mask(mask):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k, iterations=1)
    return mask

def biggest_bbox(mask, min_area=1500):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area > best_area and area >= min_area:
            best = (x, y, x + w, y + h)
            best_area = area
    return best

def main():
    lower_y, upper_y = get_yellow_limits(LEMON_YELLOW_BGR, DELTA_H, S_MIN, V_MIN)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera can't be opened.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame from the camera.")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_yellow = cv2.inRange(hsv, lower_y, upper_y)

        if EXCLUDE_SKIN:
            mask_skin = skin_mask_ycrcb(frame)
            mask = cv2.bitwise_and(mask_yellow, cv2.bitwise_not(mask_skin))
        else:
            mask = mask_yellow

        mask = clean_mask(mask)
        bbox = biggest_bbox(mask, MIN_AREA)

        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            print(f"Object found at: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        else:
            print("No object found")

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
