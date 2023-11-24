import cv2
import face_recognition
import numpy as np
import os
import json
from deepface import DeepFace
# from deepface.deepface import DeepFace

TRACK_THRESHOLD = 70
AREA_THRESHOLD = 120000

# Load images
random = face_recognition.load_image_file("known_people/random.png")

# Encodings
random_encodings = face_recognition.face_encodings(random)[0]


# Name people and encodings
people = [
    [random_encodings, "random"]
]
people_encodings = [
    random_encodings
]
people_names = [
    "random"
]

def upadate_json(face_id, image):

    # load json
    f = open('identities.json')
    data = json.load(f)
    if face_id not in data:
        try:
            features_list = DeepFace.analyze(image, enforce_detection=True)
            print(f"features list size is {len(features_list)}")
            features = features_list[0]
            age = features.get('age')
            gender = features.get('dominant_gender')
            race = features.get('dominant_race')
            # emotions = features.get('dominant_emotion')

            print("age ", age)

            data[face_id] = {
                "age": age,
                "gender": gender,
                "race": race
            }

            with open('identities.json', 'w') as outfile:
                json.dump(data, outfile)
        except:
            print("error getting attributes")
            # faces_tracked.remove(face_id)

# upadate_json("random")
# Make encodings of known people images
folder = "known_people"
def process_imgs():
    for filename in os.listdir(folder):
        if filename == ".DS_Store":
            continue
        
        process_img(filename)


def process_img(filename):
    img = face_recognition.load_image_file(f"{folder}/{filename}")
    cur_encodings = face_recognition.face_encodings(img)

    if len(cur_encodings) == 0:
        print('no encodings found')
        return
    
    if len(cur_encodings) > 0:
        cur_encodings = cur_encodings[0]

    people_encodings.append(cur_encodings)
    people_names.append(filename[:-4])
    people.append([cur_encodings, filename[:-4]])

    print(f"{folder}/{filename}")

process_imgs()


# Capture video from webcam
cap = cv2.VideoCapture(1)
i = len(people)
xc = 0
yc = 0
area = 0
center = [1920/2, 1080/2]
best_area = 185000

# Only process one frame out of every 2
process_this_frame = True
detected_faces = []


while(True):
    ret, frame = cap.read()
    
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        xc = 0
        yc = 0
        

        for (top, right, bottom, left) in face_locations:
            
            if (right-left)*(bottom-top) > best_area:
                best_area = (right-left)*(bottom-top)
                center = [(right+left)/2, (bottom+top)/2]


        # Check each encoding found
        face_names = []
        
        for face_encoding, location in zip(face_encodings, face_locations):
            # print("l____",location[0])
            

            flag = False
            # print("detected: ", len(detected_faces))
            for detected in detected_faces:
                centerx = (location[3] + (location[1] - location[3])/2)*4
                centery = (location[0] + (location[0] - location[2])/2)*4

                # print("detected: ", detected["y"], "center: ", centery, "diff: ", abs(detected["y"] - centery))

                if (abs(detected["x"] - centerx) < TRACK_THRESHOLD) and (abs(detected["y"] - centery) < TRACK_THRESHOLD):
                    name = detected["name"]
                    flag = True
                    # print("same")
                    break

                # x = 50, new x = 55, 50-55 = 5, 
            if not flag:
                name = "Unknown"

            # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(face_encoding, people_encodings, 0.6)
                

                face_distances = face_recognition.face_distance(people_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = people_names[best_match_index]
            
                
            face_names.append([name,flag])
            
    detected_faces = []
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        area = (right-left)*(bottom-top)
        # print(area)
        
    
        if process_this_frame and name[0] == "Unknown" and area > AREA_THRESHOLD:
            # print("Unknown")
            # cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            left = max(left - 50,0)
            right = right + 50
            top = max(0,top - 50)
            bottom = bottom + 50
            result = frame[top:bottom, left:right]

            # new_name = input("Enter name: ")
            # name[0] = new_name
            new_name = f"face{i}.png"
            name[0] = new_name
            new_dir = f"{folder}/{new_name}"
            cv2.imwrite(new_dir,result)
            process_img(new_name)
            upadate_json(new_name, result)
            
            # print(name[0])
            
            i = i+1


        # Draw a box around the face
        
        xc = left + (right - left)/2
        yc = top + (top - bottom)/2
        area = (right-left)*(bottom-top)
        # print(xc)

        detected_faces.append({"x": xc, "y": yc, "name": name[0]})

        # Draw a label with a name below the face
        if name[1]:
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 0, 0), cv2.FILLED)
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX

        cv2.putText(frame, name[0], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        
    process_this_frame = not process_this_frame
    
    if xc != 0:
        difx = xc - center[0] 
    
    if yc != 0:
        dify = center[1] - yc

    max_degree = 30
    

    # print(area)
    

    print(xc, ", ", yc)
   
    # Display the resulting image
    cv2.imshow('Video', frame)
    # cv2.waitKey(1)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 1080, 1920


cap.release()
cv2.destroyAllWindows()
print("Hello World!")