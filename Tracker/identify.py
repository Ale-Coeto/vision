import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "people_walking.mp4"
# cap = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture(1)


while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    # print(frame.shape[1], frame.shape[0])
    # 1280 720

    if success:
       
        # results = model.track(frame, persist=True, tracker='bytetrack.yaml', classes=0) #reid works better with bytetrack
        results = model(frame, verbose=False, classes=[56,57]) #reid works better with bytetrack

        for out in results:
            for box in out.boxes:
                x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]

                class_id = box.cls[0].item()
                label = model.names[class_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                print(label)

        

        # # Visualize the results on the frame
        # # annotated_frame = results[0].plot()

        # boxes = results[0].boxes.xywh.cpu().tolist()
        # names = results[0].names
        # # classes = results[0].classes
        # try:
        #     track_ids = results[0].boxes.id.int().cpu().tolist()
        # except Exception as e:
        #     track_ids = []

        #draw results
        # for (box, label_id, name) in zip(boxes, classes, names):
        #     x = int(box[0])
        #     y = int(box[1])
        #     w = int(box[2])
        #     h = int(box[3]) 

        #     label = "Chair" if label_id == 57 else "Couch"
        #     # print(name)
        #     cv2.rectangle(frame, (int(x - w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (255, 0, 0), 2)
        #     cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        #     # cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

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