# from ultralytics import YOLO

# # Load the model and run the tracker with a custom configuration file
# model = YOLO('yolov8n.pt')
# results = model.track(source="https://www.youtube.com/watch?v=bwJ-TNu0hGM", tracker='bytetrack.yaml')

import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "people_walking.mp4"
cap = cv2.VideoCapture(video_path)
# cap = cv2.VideoCapture(0)


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # botrack = model.BOTrack()
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        # results = model.track(frame, persist=True, tracker='bytetbytetrackrack.yaml', classes=0) #reid works better with bytetrack
        results = model.track(frame, persist=True, tracker='botsort.yaml', classes=0) #reid works better with bytetrack

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # ids = results[0].boxes.id.cpu().numpy()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()