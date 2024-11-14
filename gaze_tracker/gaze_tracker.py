import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Screen parameters for moving dot
screen_width, screen_height = 1280, 720
dot_radius = 10
speed = 3
dot_position = [screen_width // 2, screen_height // 2]

# Movement pattern (in this example, a circular path)
angle = 0
center_x, center_y = screen_width // 2, screen_height // 2
radius = 100

def update_dot_position():
    global angle, dot_position
    dot_position[0] = int(center_x + radius * np.cos(angle))
    dot_position[1] = int(center_y + radius * np.sin(angle))
    angle += 0.05  # adjust speed of the dot

def get_gaze_direction(landmarks):
    left_eye = landmarks[33]  # Example landmark for left eye
    # print(left_eye)
    right_eye = landmarks[263]  # Example landmark for right eye
    nose_tip = landmarks[1]  # Landmark for the nose

    eye_center_x = (left_eye.x + right_eye.x) / 2
    eye_center_y = (left_eye.y + right_eye.y) / 2
    eye_center_z = (left_eye.z + right_eye.z) / 2
    eye=((eye_center_x), (eye_center_y), (eye_center_z))
    
    gaze_direction = ((nose_tip.x - eye_center_x), (nose_tip.y - eye_center_y), nose_tip.z - eye_center_z)
    return gaze_direction,eye

def project_3d_to_2d(point_3d, camera_matrix):
    """
    Project 3D point to 2D using the camera matrix.
    
    Args:
        point_3d: The 3D point (x, y, z).
        camera_matrix: The camera intrinsic matrix.
    
    Returns:
        2D point (x, y) after projection.
    """
    
    point_3d_homogeneous = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
    
    point_2d_homogeneous = np.dot(camera_matrix, point_3d_homogeneous)  # Project to 2D
    
    # Convert back from homogeneous coordinates
    point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[2]
    # print(point_2d)
    return tuple(int(coord) for coord in point_2d)

def is_valid_point(point_3d):
    # Check if all coordinates of the point are zero
    return not (point_3d[0] == 0 and point_3d[1] == 0 and point_3d[2] == 0)

def draw_gaze_vector(frame, eye_center, gaze_direction, scale=1000):
    """
    Draws a gaze direction vector on the frame.

    Args:
        frame: The OpenCV frame on which to draw.
        eye_center: Tuple of (x, y) coordinates for the starting point of the vector.
        gaze_direction: Tuple of (dx, dy, dz) representing the 3D gaze direction vector.
        scale: Factor to scale the length of the gaze vector for visualization.
    """
    # Calculate the 2D end point of the gaze vector
    endpoint = (
        (eye_center[0] + gaze_direction[0] ),
        (eye_center[1] + gaze_direction[1] ),
        (eye_center[2] + gaze_direction[2] )
    )
    
    # print(f"ec{gaze_direction}")
    
    if(is_valid_point(eye_center) and is_valid_point(endpoint)):
        focal_length = 80
    
        cx, cy = 640, 360  # Center of the image (640x480)
        camera_matrix = np.array([
            [focal_length, 0, cx, 0],
            [0, focal_length, cy, 0],
            [0, 0, 1, 0]
        ], dtype=np.float32)
        camera_matrix2 = np.array([[focal_length, 0, cx], 
                          [0, focal_length, cy], 
                          [0, 0, 1]], np.float32)
        dist_coeffs = np.zeros((5, 1), np.float32) 
        rvec = np.zeros((3, 1), np.float32) 
        tvec = np.zeros((3, 1), np.float32) 
        eye_2d=project_3d_to_2d(eye_center, camera_matrix)
        endpoint_2d=project_3d_to_2d(endpoint, camera_matrix)
        
        points_2d, _ = cv2.projectPoints(eye_center, 
                                 rvec, tvec, 
                                 camera_matrix2, 
                                 dist_coeffs)
        point_tuple = tuple(points_2d[0][0])
        point_tuple = (int(point_tuple[0]), int(point_tuple[1]))
        # Optionally, draw the starting point as a small circle
        print(endpoint_2d)
        print(point_tuple)
        eye_center = ((eye_center[0]*1280),(eye_center[1]*720),eye_center[2]*1)
        endpoint = ((endpoint[0]*1280),(endpoint[1]*720),endpoint[2]*1)
        # Draw a line from eye center to the endpoint (gaze direction)
        cv2.line(frame, (int(eye_center[0]),int(eye_center[1])), (int(endpoint[0]),int(endpoint[1])), (0, 255, 0), 5)  # Green color, 2px thickness
        # cv2.circle(frame, point_tuple, 5, (0, 255, 0), -1)
        cv2.circle(frame, (int(eye_center[0]),int(eye_center[1])), 3, (0, 0, 255), -1)  # Red color, filled circle
        cv2.line(frame, (int(eye_center[0]),int(eye_center[1])), endpoint_2d,  (0, 0, 255), 5)
# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, screen_width)
cap.set(4, screen_height)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally for a mirror view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Update dot position
    update_dot_position()

    # Draw the dot on screen
    cv2.circle(frame, (dot_position[0], dot_position[1]), dot_radius, (0, 0, 255), -1)

    # Detect face landmarks
    result = face_mesh.process(rgb_frame)
    if result.multi_face_landmarks:
        for facial_landmarks in result.multi_face_landmarks:
            # Draw facial landmarks on the frame
            mp_drawing.draw_landmarks(
                frame, facial_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=1)
            )

            # Calculate gaze direction
            landmarks = facial_landmarks.landmark
            gaze_direction,eye = get_gaze_direction(landmarks)

            # Check if the user’s gaze is following the dot (this is a simplified example)
            # dot_vector = (dot_position[0] - screen_width // 2, dot_position[1] - screen_height // 2)
            # gaze_vector = (gaze_direction[0] * screen_width, gaze_direction[1] * screen_height)
            # similarity = np.dot(dot_vector, gaze_vector) / (np.linalg.norm(dot_vector) * np.linalg.norm(gaze_vector) + 1e-6)

            # # If similarity is above threshold, we assume the gaze follows the dot
            # if similarity > 0.8:
            #     cv2.putText(frame, "Gaze Detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # else:
            #     cv2.putText(frame, "Gaze Not Detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    draw_gaze_vector(frame, eye, gaze_direction)
            
            
    # Display the frame
    cv2.imshow("Gaze Detection", frame)

    # Break on 'q' key press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
