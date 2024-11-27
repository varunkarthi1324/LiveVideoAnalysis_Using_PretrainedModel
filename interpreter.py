import cv2
import mediapipe as mp
from collections import deque

# Initialize Mediapipe Hands and Face Mesh
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.9)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.9, min_tracking_confidence=0.9)

# Finger tip landmark indices and trail color
finger_tips = [4, 8, 12, 16, 20]
trail_color = (255, 255, 0)  # Yellow trails for hand movement

# Face landmark indices to track (e.g., nose tip and chin)
face_points = {"Nose": 1, "Chin": 152}
face_color = (0, 0, 255)  # Red trails for face movement

# Initialize deque to store trails for hands and face
trails = {
    "Left": {idx: deque(maxlen=50) for idx in finger_tips},
    "Right": {idx: deque(maxlen=50) for idx in finger_tips},
    "Face": {name: deque(maxlen=50) for name in face_points}
}

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip image horizontally for natural view
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks with Mediapipe
    hand_results = hands.process(frame_rgb)
    # Process face landmarks with Mediapipe
    face_results = face_mesh.process(frame_rgb)

    # Hands Tracking
    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            # Determine hand label (Left/Right)
            label = handedness.classification[0].label

            # Draw the joints and connections (lines)
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),  # Red joints
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)  # Green lines
            )

            # Update and draw trails for finger tips
            for idx in finger_tips:
                x = int(hand_landmarks.landmark[idx].x * frame.shape[1])
                y = int(hand_landmarks.landmark[idx].y * frame.shape[0])

                # Append current position to trail deque
                trails[label][idx].append((x, y))

                # Draw the trail
                for i in range(1, len(trails[label][idx])):
                    if trails[label][idx][i - 1] and trails[label][idx][i]:
                        cv2.line(frame, trails[label][idx][i - 1], trails[label][idx][i], trail_color, 2)

            # Calculate and draw a bounding box around the hand
            x_min = min([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1]
            x_max = max([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1]
            y_min = min([landmark.y for landmark in hand_landmarks.landmark]) * frame.shape[0]
            y_max = max([landmark.y for landmark in hand_landmarks.landmark]) * frame.shape[0]

            # Draw bounding box
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)  # Blue box
            cv2.putText(frame, f"{label} Hand", (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Face Movement Tracking with Mesh
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            ih, iw, _ = frame.shape

            # Get bounding box for the face
            x_min = min([landmark.x for landmark in face_landmarks.landmark]) * iw
            x_max = max([landmark.x for landmark in face_landmarks.landmark]) * iw
            y_min = min([landmark.y for landmark in face_landmarks.landmark]) * ih
            y_max = max([landmark.y for landmark in face_landmarks.landmark]) * ih

            # Draw bounding box around the face
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)  # Green box

            # Update and draw trails for specific face points
            for name, idx in face_points.items():
                x = int(face_landmarks.landmark[idx].x * iw)
                y = int(face_landmarks.landmark[idx].y * ih)

                # Append current position to trail deque
                trails["Face"][name].append((x, y))

                # Draw the trail
                for i in range(1, len(trails["Face"][name])):
                    if trails["Face"][name][i - 1] and trails["Face"][name][i]:
                        cv2.line(frame, trails["Face"][name][i - 1], trails["Face"][name][i], face_color, 2)

                # Draw the landmark position
                cv2.circle(frame, (x, y), 4, face_color, -1)
                cv2.putText(frame, name, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 1)

            # Draw the face mesh lines
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),  # Red mesh
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)  # Green lines
            )

    # Display the frame
    cv2.imshow("Hand and Face Movement Tracker", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
