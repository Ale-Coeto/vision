from PIL import Image
import cv2
from ultralytics import YOLO
from reid_model import check_visibility, load_network, compare_images, extract_feature_from_img, is_full_body, get_structure
import torch.nn as nn
import torch
import tqdm
import mediapipe as mp

pbar = tqdm.tqdm(total=5, desc="Loading models")

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')
pbar.update(1)

# Load media pipe model
pose_model = mp.solutions.pose.Pose(min_detection_confidence=0.8) 
pbar.update(1)

# Load the ReID model
structure = get_structure()
pbar.update(1)
model_reid = load_network(structure)
pbar.update(1)
model_reid.classifier.classifier = nn.Sequential()
pbar.update(1)
use_gpu = torch.cuda.is_available()
if use_gpu:
    model_reid = model_reid.cuda()
pbar.close()

# Open the video file or use camera
USE_CAMERA = True
THRESHOLD = 1
if USE_CAMERA:
    cap = cv2.VideoCapture(1)
else:
    video_path = "cp_2.mp4"
    cap = cv2.VideoCapture(video_path)

# Initialize lists to store information about people
people_tags = []
people_ids = []
people_features = []
prev_ids = []

# Loop through the video frames
while cap.isOpened():

    # Read a frame from the video
    success, frame = cap.read()

    if success:
        width = frame.shape[1]
        
        # Get the results from the YOLOv8 model
        results = model.track(frame, persist=True, tracker='bytetrack.yaml', classes=0, verbose=False) #could use botsort.yaml
        
        # Get the bounding boxes and track ids
        boxes = results[0].boxes.xywh.cpu().tolist()
        track_ids = []

        try:
            track_ids = results[0].boxes.id.int().cpu().tolist()
        except Exception as e:
            track_ids = []

        false_detections = []
        # Check if there is a new id
        for (box, track_id) in zip(boxes, track_ids):
            if track_id not in prev_ids:

                # Get bbox
                x = int(box[0])
                y = int(box[1])
                w = int(box[2])
                h = int(box[3]) 
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)

                # Crop the image 
                cropped_image = frame[y1:y2, x1:x2]
                pil_image = Image.fromarray(cropped_image)
                # person = check_visibility(pose_model,cropped_image)
                person = True
                # print("person", person)

                if not person or x1 <= THRESHOLD or x2 >= width - THRESHOLD:
                    false_detections.append(track_id)
                    continue

                # cv2.imshow('crpd',cropped_image)

                # Get feature
                with torch.no_grad():
                    new_feature = extract_feature_from_img(pil_image, model_reid)
                flag = False

                # Check if there is a match with seen people
                for i, person_feature in enumerate(people_features):
                    # if person_feature is None:
                    #     continue
                    #Compare features
                    match = compare_images(person_feature, new_feature)

                    # If there is a match and the person matched is not currently in the frame (shouldnt be two people with the same id in the same frame)
                    if match and people_ids[i] not in track_ids:

                        # Update id to the id assigned by yolo
                        people_ids[i] = track_id
                        flag = True
                        break
                
                # If there is no match and the id is not already saved
                if not flag and track_id not in people_ids:
                    print("New person detected")
                    people_ids.append(track_id)
                    people_tags.append(f"Person {len(people_ids)}")
                    people_features.append(new_feature)
                
        print(track_ids)
        print(people_tags)
        print(people_ids)
        prev_ids = []
        # Draw results
        for (box, track_id) in zip(boxes, track_ids):
            if track_id in false_detections:
                continue

            prev_ids.append(track_id)

            x = int(box[0])
            y = int(box[1])
            w = int(box[2])
            h = int(box[3]) 
            id = people_ids.index(track_id)
            name = people_tags[id]

            cv2.rectangle(frame, (int(x - w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (255, 0, 0), 2)
            cv2.putText(frame, name, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA) # Tag
            cv2.putText(frame, str(track_id), (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA) # Yolo ID

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", frame)
        # prev_ids = track_ids

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # End of video
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()