import face_recognition
import cv2
import numpy as np


# Load images
ale = face_recognition.load_image_file("Ale.png")
test_img = face_recognition.load_image_file("unknown_people/Test2.jpg")

# Encodings
ale_encodings = face_recognition.face_encodings(ale)[0]

# Name people and encodings
people = [
    [ale_encodings, "ale"]
]
people_encodings = [
    ale_encodings
]
people_names = [
    "ale"
]

# # Find people locations in img
unknown_locations = face_recognition.face_locations(test_img)
for detected in unknown_locations:
    t,r,b,l = detected
    cv2.rectangle(test_img, (l,t),(r,b),(0,255,0), 2)

# Recognize people
unknown_encodings = face_recognition.face_encodings(test_img, unknown_locations)

for encoding, name in people:
    match = face_recognition.compare_faces(unknown_encodings, encoding, 0.55)
    if True in match:
        print(name)
        index = match.index(True)
        
        t,r,b,l = unknown_locations[index]
        test_img = cv2.rectangle(test_img, (l,t),(r,b),(255,0,0), 2)
        test_img = cv2.putText(test_img, name, (l+4,b-4), 1, 2,(255,0,0), 2)

# for encoding in unknown_encodings:
#     matches = face_recognition.compare_faces(people_encodings, encoding)
#     print(matches)

# for encoding in unknown_encodings:
#     matches = face_recognition.compare_faces(people_encodings, encoding)
#     if True in matches:
#         first_match_index = matches.index(True)
#         name = people_names[first_match_index]
#         print(name)

    # name = "unknwo"
    # face_distances = face_recognition.face_distance(people_encodings, encoding)
    # best_match_index = np.argmin(face_distances)
    # if matches[best_match_index]:
    #     name = people_names[best_match_index]
    #     print(name)


test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGBA)
cv2.imshow("found", test_img)
cv2.waitKey(0)
