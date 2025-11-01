import cv2
import mediapipe as mp

# import image data
path = '/Users/damianhuckele/Developer/Python/ComputerVision/face_recognition_and_anonymization/img.jpg'
img = cv2.imread(path)

H, W, _ = img.shape

# detect faces in image
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection = 0, min_detection_confidence = 0.5) as face_detection:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    print(out.detections)

    if out.detections is not None:
        for val in out.detections:
            loc_data = val.location_data
            bbox = loc_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            cv2.rectangle(img, (x1,y1), (x1+w, y1+h), (0,255,0), 2)

        cv2.imshow('img', img)
        cv2.waitKey(0)
    else:
        print("Image does not contain face.")