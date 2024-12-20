from ultralytics import YOLO
import cv2

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv, write_to_csv
# from visualize import *


results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('best.pt')

# load video
cap = cv2.VideoCapture('sample.mp4')

vehicles = [2, 3, 5, 7]

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img
# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            try:
                if car_id != -1:

                    # crop license plate
                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                    
                    # process license plate
                    print(license_plate_crop.shape)
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    # read license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                    if license_plate_text is not None:
                        results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                    'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': license_plate_text,
                                                                        'bbox_score': score,
                                                                        'text_score': license_plate_text_score}}
                        data = {
                            'frame_nmr': frame_nmr,
                            'car_id': car_id,
                            'bbox': [xcar1, ycar1, xcar2, ycar2],  #Example bounding box coordinates
                            'license_plate': [x1, y1, x2, y2], # Example bounding box coordinates
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_score
                        }
                        write_to_csv(data, 'license_plate_data.csv')
                        draw_border(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0), 25,
                            line_length_x=200, line_length_y=200)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)
                        license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                        license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))
                        H, W, _ = license_crop.shape
                        try:
                            frame[int(ycar1) - H - 100:int(ycar1) - 100,
                            int((xcar2 + xcar1 - W) / 2):int((xcar2 + xcar1 + W) / 2), :] = license_crop

                            frame[int(ycar1) - H - 400:int(ycar1) - H - 100,
                            int((xcar2 + xcar1 - W) / 2):int((xcar2 + xcar1 + W) / 2), :] = (255, 255, 255)

                            (text_width, text_height), _ = cv2.getTextSize(license_plate_text,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                4.3,
                                    17)

                            cv2.putText(frame,
                                        license_plate_text,
                                        (int((xcar2 + xcar1 - text_width) / 2), int(ycar1 - H - 250 + (text_height / 2))),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        4.3,
                                        (0, 0, 0),
                                        17)

                        except:
                            pass
                        frame = cv2.resize(frame, (1280, 720))

                        cv2.imshow('frame', frame)
                        # Break the loop if 'q' is pressed
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            # write results
                            # write_csv(results, './test1.csv')
                            break
            except:
                pass

# write results
# write_csv(results, './test1.csv')

                    