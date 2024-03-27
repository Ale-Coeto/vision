# from ultralytics import YOLO
# pip install -U ultralytics
# # Load the model and run the tracker with a custom configuration file
# model = YOLO('yolov8n.pt')
# results = model.track(source="https://www.youtube.com/watch?v=bwJ-TNu0hGM", tracker='bytetrack.yaml')

import cv2
from ultralytics import YOLO
from reid_model import load_network, compare_images, extract_feature_from_img, get_structure
import torch.nn as nn

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "people_walking.mp4"
cap = cv2.VideoCapture(video_path)
# cap = cv2.VideoCapture(0)

people_tags = []
people_ids = []
people_features = []

structure = get_structure()
model_reid = load_network(structure)
model_reid.classifier.classifier = nn.Sequential()

use_gpu = True
if use_gpu:
    model_reid = model_reid.cuda()

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    # print(frame.shape[1], frame.shape[0])
    # 1280 720

    if success:
       
        # results = model.track(frame, persist=True, tracker='bytetrack.yaml', classes=0) #reid works better with bytetrack
        results = model.track(frame, persist=True, tracker='botsort.yaml', classes=0, verbose=False) #reid works better with bytetrack

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        boxes = results[0].boxes.xywh.cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for (box, track_id) in zip(boxes, track_ids):
            if track_id not in people_ids:
                # print(track_id)
                # get feature
                x = int(box[0])
                y = int(box[1])
                w = int(box[2])
                h = int(box[3]) 
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)

                # Crop the image using the calculated coordinates
                cropped_image = frame[y1:y2, x1:x2]
                # new_feature = 0
                with torch.no_grad():
                    new_feature = extract_feature_from_img(cropped_image, model_reid)
                flag = False

                for i, person_feature in enumerate(people_features):
                    #Compare features
                    # match = True
                    match = compare_images(person_feature, new_feature)
                    # ans = input("Match? (y/n): ")
                    # match = ans == 'y'

                    if match and people_ids[i] not in track_ids:
                        # Update id
                        people_ids[i] = track_id
                        flag = True
                        break

                if flag == False:
                    people_ids.append(track_id)
                    people_tags.append(f"Person {track_id}")
                    people_features.append(new_feature)
                
        print(people_ids)
        #draw results
        for (box, track_id) in zip(boxes, track_ids):
            x = int(box[0])
            y = int(box[1])
            w = int(box[2])
            h = int(box[3]) 
            id = people_ids.index(track_id)
            name = people_tags[id]

            cv2.rectangle(frame, (int(x - w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (255, 0, 0), 2)
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # print(people_ids.tolist())
        # print(track_ids)
        # print(boxes[0])
        # print(track_ids[0])
        # # ids = results[0].boxes.id.cpu().numpy()
        # x = int(boxes[0][0])
        # y = int(boxes[0][1])
        # w = int(boxes[0][2])
        # h = int(boxes[0][3])

        # cv2.rectangle(frame, (int(x - w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (255, 0, 0), 2)
        # # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()