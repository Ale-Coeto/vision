import mediapipe as mp
import cv2

# Calling the pose solution from MediaPipe
mp_pose = mp.solutions.pose

images = 37

# Opening the image source to be used
image = cv2.imread("image.jpg")

# Calling the pose detection model
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    # Detecting the pose with the image
    poseResult = pose.process(image)