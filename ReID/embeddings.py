from PIL import Image
import cv2
from ultralytics import YOLO
from reid_model import extract_feature_from_path, check_visibility, load_network, compare_images, extract_feature_from_img, is_full_body, get_structure
import torch.nn as nn
import torch
import os
import numpy as np


def track_video(model, reid_model, filevideo):
        """
        This method run a tracker from ultralytics given a detector model and
        a video.
        Args:
            filevideo[str]: path to the video to track
            model: model of the detector that the tracker will use
            time: timestamp from the camera entity record
        Returns: 
            [dict]: a dictionary with person entities detected from the video
        """


        out_path = os.path.join('test_embeddings','1'+'_{}.npy'.format("dense"))
        if not os.path.exists("test_embeddings"):
            os.makedirs("test_embeddings")

        emb = []

        # print(filevideo)
        video = cv2.VideoCapture(filevideo)
        frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        first_person = False    
        entities = {}
        print(frames)
        
        
        for i_frame in range(20):
            ret, frame = video.read()
            if not ret:
                continue
            # if i_frame >= 10 and i_frame <= 90:
            #     continue

            # Track all objects in the frame
            results = model.track(source=frame, persist=True, classes=0, show=False, verbose=False)
            names = results[0].names
            boxes = results[0].boxes
            bboxes = results[0].boxes.xywh.cpu().tolist()
            
            # Get the timestamp of the frame
            # seconds = i_frame / video.get(cv2.CAP_PROP_FPS)
            # timestamp = datetime.timestamp(time + timedelta(seconds=seconds))
            # Iterate over all detected objects
            # for i, id in enumerate(boxes.id.int().tolist()):
            #     if id not in entities:
            #         entities[id] = dict(classes=[], bboxes=[], coordinates=[], timestamps=[])

            print(i_frame)
            if not boxes.is_track:
                continue

            # Features from the first person detected
            if first_person == False:
                x = int(bboxes[0][0])
                y = int(bboxes[0][1])
                w = int(bboxes[0][2])
                h = int(bboxes[0][3]) 
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)
                cropped_image = frame[y1:y2, x1:x2]
                img = Image.fromarray(cropped_image)
                cv2.imwrite("first_person.jpg", cropped_image)
                first_person = True
                print('found')

                with torch.no_grad():
                    first_feature = extract_feature_from_img(img, reid_model)
                    print("First person")

            

            # Features from the last person detected
            x = int(bboxes[0][0])
            y = int(bboxes[0][1])
            w = int(bboxes[0][2])
            h = int(bboxes[0][3]) 
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            cropped_image = frame[y1:y2, x1:x2]
            img = Image.fromarray(cropped_image)
            # cv2.imshow('A',img)
            cv2.imwrite("last_person.jpg", cropped_image)

            with torch.no_grad():
                last_feature = extract_feature_from_img(img, reid_model)
                emb.append(last_feature)
                print("Las person")
            
            # _class = names[boxes.cls[i].int().tolist()]
            # bbox = boxes.xyxy[i].numpy().astype(float).reshape((2, 2))
            # # Get the coordinates of the bbox
            # coord = bbox_to_point(bbox)
            # # TODO: Fix weirdness with homography and coordinates
            # # Explaination: The homography is saving the calibration info as [lat, lon] and therefore we need to adhere to the convention by swapping the coordinates.
            # # bbox xyxy -> point coord [x, y] -> [lat, lon]
            # coord = [1 - (coord[1] / height), coord[0] / width]
            # entities[id]['classes'].append(_class)
            # entities[id]['bboxes'].append(bbox)
            # entities[id]['coordinates'].append(coord)
            # entities[id]['timestamps'].append(timestamp)
                
        match = compare_images(first_feature, last_feature)
        print(match)
        # first_feature = extract_feature_from_path('first_person.jpg', model)
        # last_feature = extract_feature_from_path('last_person.jpg', model)
        # match = compare_images(first_feature, last_feature)
        if match:
            print("Match")  
        else:
            print("No match")

        embeddings = np.array(emb)
        np.save(out_path,embeddings)
            

        # TODO: Infer activity instead of assigning a random activity
        # random_activity = ACTIVITIES[int(random.random() * len(ACTIVITIES))]

        
        # Get the most common class for each entity
        # entities = {
        #     k: dict(
        #         **v,
        #         _class=max(v['classes'], key=v['classes'].count),
        #         activity=random_activity
        #     )
        #     for k, v in entities.items()
        #            }
        
        # return entities


model = YOLO('yolov8n.pt')

structure = get_structure()
model_reid = load_network(structure)
model_reid.classifier.classifier = nn.Sequential()

use_gpu = torch.cuda.is_available()
if use_gpu:
    model_reid = model_reid.cuda()

video_path = "cp.mp4"
track_video(model, model_reid, video_path)
