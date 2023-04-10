import cv2
import face_recognition
import os

# Load known face encodings from images in a folder
known_faces = []
known_face_names = []
known_faces_dir = "C:/Himanshu/FaceRecognition/dataset1/img"
for filename in os.listdir(known_faces_dir):
    image_path = os.path.join(known_faces_dir, filename)
    image = face_recognition.load_image_file(image_path)
    face_encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(face_encoding)
    known_face_names.append(os.path.splitext(filename)[0])

# Open the video file
cap = cv2.VideoCapture("C:/Himanshu/FaceRecognition/dataset1/Video_clip/VID_20230302_170157 (online-video-cutter.com).mp4")
face_locations = []

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    
    if not ret:
        # Break the loop if the end of the video is reached
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
    
    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        # Compare the face encoding with known face encodings
        matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)
        name = "Unknown"
        if True in matches:
            # Find the first known face that matches and use its name
            idx = matches.index(True)
            name = known_face_names[idx]
        face_names.append(name)

    # Draw bounding boxes and labels around the faces
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
        # Draw a label with the name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    cv2.waitKey(1)
    
    # Press 'q' to stop
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video file and close all windows
cap.release()
cv2.destroyAllWindows()
