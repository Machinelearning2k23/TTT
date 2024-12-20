from ultralytics import YOLO
import cv2
import json
import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv, write_to_csv
#import jsonlines
# from visualize import *

results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('best.pt')

vehicles = [2, 3, 5, 7]

# Specify the file name
file_path = './license_plate_data_.json'


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


def detect_license_plates(frame,frame_nmr):
    frames_dir="output_frames"
    os.makedirs(frames_dir, exist_ok=True)
    
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
        # json_data_gen=[]
        x1, y1, x2, y2, score, class_id = license_plate
        print(x1, y1, x2, y2)

        # assign license plate to car
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
        
        print(xcar1, ycar1, xcar2, ycar2)
        try:
            if car_id != -1:
                
                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                
                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                print("license_plate_text: ",license_plate_text)
                if license_plate_text is not None:
                    
                    data = {
                        'frame_nmr': frame_nmr,
                        'car_id': car_id,
                        'bbox': [xcar1, ycar1, xcar2, ycar2],  
                        'license_plate': [x1, y1, x2, y2], 
                        'text': license_plate_text,
                        'bbox_score': score,
                        'text_score': license_plate_text_score
                    }
                    # print("data: ",data)
                    # json_data = json.dumps(data) 
                    print("json_data: ", data)
                    # json_data_gen.append(data)
                    try:
                        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                            with open(file_path, 'r+', encoding='utf-8') as f:
                                f.seek(0)  # Go to the beginning of the file
                                content = f.read()
                                f.seek(0)  # Go back to the beginning
                                f.truncate() # Clear the file contents (important!)

                                # Remove the last ] and add a comma
                                updated_content = content[:-1] + ','

                                f.write(updated_content)
                                json.dump(data, f, indent=4)
                                f.write(']') # Add back the closing ]          # Add back ]
                        else:  # File is new or empty, create the array
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write('[')
                                json.dump(data, f, indent=4)
                                f.write(']')
                        # with jsonlines.open(file_path, mode='a') as writer:
                        #     writer.write(data)
                    except Exception as e:
                        print(f"Error appending to JSON array: {e}")

                    draw_border(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0), 25,
                            line_length_x=200, line_length_y=200)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)
                    license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                    license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))
                    H, W, _ = license_crop.shape
                    print("H W ",H, W )
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
                
                    # frame = cv2.resize(frame, (1280, 720))
                    frame_path = os.path.join(frames_dir, f"frame_{frame_nmr:04d}.png")
                    cv2.imwrite(frame_path, frame)
                    return frame, license_plate_text
        except:
            pass

    # try:
    #     existing_data=[]
    #     if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
    #         with open(file_path, 'r') as f:
    #                 existing_data = json.load(f)
    #     if json_data_gen:
    #         with open(file_path, 'w') as f:
    #             json.dump(existing_data.append(json_data_gen),f,indent=4)
    #             print(f"JSON data updated in: {file_path}")

    # except Exception as e:
    #     print(f"Error updating JSON file: {e}")
    return frame, ""
