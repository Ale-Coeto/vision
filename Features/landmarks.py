from PIL import Image, ImageDraw
import face_recognition

# Load the jpg file into a numpy array
image = face_recognition.load_image_file("random.png")

# Find all facial features in all the faces in the image
face_landmarks_list = face_recognition.face_landmarks(image)

print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

# Create a PIL imagedraw object so we can draw on the picture
pil_image = Image.fromarray(image)
d = ImageDraw.Draw(pil_image)


for face_landmarks in face_landmarks_list:
    # print("The {} in this face has the following points: {}".format("lip top", face_landmarks["top_lip"]))
    # print("The {} in this face has the following points: {}".format("lip bottom", face_landmarks["bottom_lip"]))
    sorted_list_up = sorted(face_landmarks["top_lip"], key=lambda x: x[0])
    sorted_list_down = sorted(face_landmarks["bottom_lip"], key=lambda x: x[0])


    center_up = sorted_list_up[face_landmarks["top_lip"].__len__()//2]
    center_down = sorted_list_down[face_landmarks["bottom_lip"].__len__()//2]
    # print("Center up: {}".format(center_up))
    # print("Center bottom: {}".format(center_down))

    difference = abs(center_up[1] - center_down[1])
    print("Difference: {}".format(difference))
    if difference > 10:
        print("Open mouth")

    #Draw a point
    d.point(center_up, fill=(255,0,0))
    d.point(center_down, fill=(255,0,0))
   
   
   # Print the location of each facial feature in this image
    # for facial_feature in face_landmarks.keys():
    #     print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

    # Let's trace out each facial feature in the image with a line!
    # d.line(face_landmarks["top_lip"], width=2)
    # d.line(face_landmarks["bottom_lip"], width=2)    
    # for facial_feature in face_landmarks.keys():
    #     d.line(face_landmarks[facial_feature], width=5)

# Show the picture
pil_image.show()