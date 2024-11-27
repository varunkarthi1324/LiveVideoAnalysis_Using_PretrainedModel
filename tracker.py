import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Initialize variables for shapes
shapes = ["Square", "Rectangle", "Circle", "Pentagon"]
selected_shape = None
shapes_positions = []  # Store positions, types, and sizes of drawn shapes
dragging_shape_index = None  # Track which shape is being dragged
dragging = False  # Track if a shape is being dragged

# Trash area position
trash_area = (20, 400, 120, 450)  # x1, y1, x2, y2

# Function to draw a square
def draw_square(frame, center, size, color):
    top_left = (int(center[0] - size // 2), int(center[1] - size // 2))
    bottom_right = (int(center[0] + size // 2), int(center[1] + size // 2))
    cv2.rectangle(frame, top_left, bottom_right, color, -1)

# Function to draw a rectangle
def draw_rectangle(frame, center, width, height, color):
    top_left = (int(center[0] - width // 2), int(center[1] - height // 2))
    bottom_right = (int(center[0] + width // 2), int(center[1] + height // 2))
    cv2.rectangle(frame, top_left, bottom_right, color, -1)

# Function to draw a circle
def draw_circle(frame, center, radius, color):
    cv2.circle(frame, center, int(radius), color, -1)

# Function to draw a pentagon
def draw_pentagon(frame, center, size, color):
    points = []
    for i in range(5):
        angle = np.radians(72 * i - 90)  # 360/5 and start from top
        x = int(center[0] + size * np.cos(angle))
        y = int(center[1] + size * np.sin(angle))
        points.append((x, y))
    cv2.fillPoly(frame, [np.array(points)], color)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert to RGB
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame for hand detection
    results = hands.process(frame_rgb)

    # Draw the trash area
    cv2.rectangle(frame, (trash_area[0], trash_area[1]), (trash_area[2], trash_area[3]), (0, 0, 255), 2)
    cv2.putText(frame, "Trash", (trash_area[0] + 5, trash_area[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Draw shape selection options
    for i, shape in enumerate(shapes):
        position = (50, 50 + i * 50)
        color = (0, 255, 0) if shape == selected_shape else (255, 255, 255)
        cv2.putText(frame, shape, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(frame, (position[0] - 40, position[1] - 20),
                      (position[0] + 60, position[1] + 10), (200, 200, 200), 1)

    # Handle hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get thumb and index finger positions
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_x, thumb_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
            index_x, index_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])

            # Highlight index finger
            cv2.circle(frame, (index_x, index_y), 10, (0, 0, 255), -1)

            # Check for selection of shapes
            for i, shape in enumerate(shapes):
                shape_x, shape_y = 50, 50 + i * 50
                if abs(index_x - shape_x) < 50 and abs(index_y - shape_y) < 25:
                    selected_shape = shape

            # Check for dragging existing shapes
            if not dragging:
                for i, (shape_type, shape_pos, shape_size) in enumerate(shapes_positions):
                    x, y = shape_pos
                    if abs(index_x - x) < shape_size and abs(index_y - y) < shape_size:
                        if abs(thumb_x - index_x) < 30 and abs(thumb_y - index_y) < 30:  # Pinch gesture
                            dragging_shape_index = i
                            dragging = True
                            break

            # Update shape position if dragging
            if dragging and dragging_shape_index is not None:
                shapes_positions[dragging_shape_index][1] = (index_x, index_y)

                # Check if shape is dragged into the trash area
                if trash_area[0] <= index_x <= trash_area[2] and trash_area[1] <= index_y <= trash_area[3]:
                    shapes_positions.pop(dragging_shape_index)
                    dragging_shape_index = None
                    dragging = False

            # Release shape when pinch ends
            if dragging and (abs(thumb_x - index_x) > 30 or abs(thumb_y - index_y) > 30):
                dragging = False
                dragging_shape_index = None

            # Add new shape on tap gesture (thumb and index close together)
            if selected_shape and abs(thumb_x - index_x) < 30 and abs(thumb_y - index_y) < 30:
                shapes_positions.append([selected_shape, (index_x, index_y), 50])
                selected_shape = None  # Reset selection

    # Draw shapes
    for i, (shape_type, shape_pos, shape_size) in enumerate(shapes_positions):
        color = (0, 255, 255)
        if shape_type == "Square":
            draw_square(frame, shape_pos, shape_size, color)
        elif shape_type == "Rectangle":
            draw_rectangle(frame, shape_pos, shape_size * 1.5, shape_size, color)
        elif shape_type == "Circle":
            draw_circle(frame, shape_pos, shape_size // 2, color)
        elif shape_type == "Pentagon":
            draw_pentagon(frame, shape_pos, shape_size, color)

    # Display the frame
    cv2.imshow("Shape Drawer", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
