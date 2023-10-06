import mediapipe as mp
import cv2
import os

# Calling the pose solution from MediaPipe
mp_pose = mp.solutions.pose

# Calling the solution for image drawing from MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# Opening the image source to be used
images = 37

images_dir = "raw"
output_dir = "processed"
# Calling the pose detection model
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    # Detecting the pose with the image
    for i in range(1, images + 1):
        print("image ", i)
        print(f"{images_dir}/NAO{i}.jpeg")
        print(os.path.exists(f"{images_dir}/NAO{i}.jpeg"))
        image = cv2.imread(f"{images_dir}/NAO{i}.jpeg")
        image_height, image_width, _ = image.shape
        # Converting the BGR image to RGB
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # Drawing the pose landmarks on the image
        if(results.pose_landmarks is None):
            print("No se detecto nada")
            continue
        annotated_image = image.copy()
        annotated_image.flags.writeable = True
        # annotated_image.flags.writeable = True
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        print(results.pose_landmarks)
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        # cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        cv2.imwrite(f"{output_dir}/image{i}.jpg", annotated_image)
        # Printing the pose landmarks
        print(f"Landmarks of image{i}.jpg:")
        print(results.pose_landmarks)
        print()