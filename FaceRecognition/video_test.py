import cv2
import face_recognition
import numpy as np

# Load images
ale = face_recognition.load_image_file("Ale.png")

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

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('/Users/riddhamanna/Documents/RoboDK/ymaze.mp4')
process_this_frame = True

while(True):
    ret, img = cap.read()
    
    if process_this_frame:
        small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        #rgb_small_frame = small_frame[:, :, ::-1]

        unknown_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, unknown_locations)

        for detected in unknown_locations:
            t,r,b,l = detected
            t *= 4
            r *= 4
            b *= 4
            l *= 4
            cv2.rectangle(img, (l,t),(r,b),(0,255,0), 2)

            
        

        for encoding, name in people:
            match = face_recognition.compare_faces(face_encodings, encoding, 0.55)
            if True in match:
                #print(name)
                index = match.index(True)
                
                t,r,b,l = unknown_locations[index]
                t *= 4
                r *= 4
                b *= 4
                l *= 4
                img = cv2.rectangle(img, (l,t),(r,b),(255,0,0), 2)
                img = cv2.putText(img, name, (l+4,b-4), 1, 2,(255,0,0), 2)

     

        cv2.imshow('frame',img)
        if cv2.waitKey(1) & 0xFF == 27:#ord('q'):
            break

    process_this_frame = not process_this_frame

cap.release()
cv2.destroyAllWindows()
print("Hello World!")