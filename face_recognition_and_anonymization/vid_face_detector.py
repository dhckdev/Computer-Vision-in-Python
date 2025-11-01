import cv2
import mediapipe as mp
import argparse

def image_process(img, face_detection):
    H, W, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for val in out.detections:
            loc_data = val.location_data
            bbox = loc_data.relative_bounding_box
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

    return img

# create arguments for mode and file location
args = argparse.ArgumentParser()
args.add_argument('--mode', default='webcam')
args.add_argument('--filePath', default=None)
args = args.parse_args()

# detect faces in image
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection = 0, min_detection_confidence = 0.5) as face_detection:

    if args.mode in ['image']:
        img = cv2.imread(args.filePath)
        H, W, _ = img.shape
        img = image_process(img, face_detection)

        cv2.imshow('img', img)
        cv2.waitKey(0)

    elif args.mode in ['webcam']:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()

        while ret:
            frame = image_process(frame, face_detection)
            cv2.imshow('frame', frame)
            cv2.waitKey(10)
            ret, frame = cap.read()

            # listen for key interrupt
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # release memory and close windows
        cap.release()
        cv2.destroyAllWindows()
