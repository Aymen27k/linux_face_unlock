import os
import time
import glob
import warnings
import cv2
import face_recognition


warnings.filterwarnings( "ignore", message="pkg_resources is deprecated as an API" )

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "faces/aymen/aymen*.jpg")
UNLOCK_COMMAND = "loginctl unlock-session"
GREET_COMMAND = 'espeak "Hello Aymen"'
MOTION_THRESHOLD = 20
SCALE_FACTOR = 0.25  # Processing at 1/4 size for speed
SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

def load_known_faces(path_pattern):
    """Loads images from a path and returns encodings and names."""
    encodings = []
    names = []
    print(f"Loading known faces from {path_pattern}...")
    
    for file in glob.glob(path_pattern):
        image = face_recognition.load_image_file(file)

        face_encs = face_recognition.face_encodings(image)
        if face_encs:
            encodings.append(face_encs[0])
            names.append("Aymen")
            
    print(f"Loaded {len(encodings)} reference encodings.")
    return encodings, names

def detect_motion(f1, f2):
    """Calculates if there is enough movement between two frames."""
    diff = cv2.absdiff(f1, f2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours) > 0

def recognize_faces(frame, known_encodings, known_names):
    """Detects and identifies faces in a frame."""
    # Pre-process frame (Resize and Color Convert)
    small_frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    
    if not face_locations:
        return None, []

    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    detected_names = []
    for face_encoding in face_encodings:
        # compare_faces returns a list of Booleans
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
        
        detected_names.append(name)

    return face_locations, detected_names

def main():
    # 1. Setup
    known_encs, known_names = load_known_faces(KNOWN_FACES_DIR)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    unlocked = False
    snapshot_taken = False
    last_motion_time = None
    MOTION_TIMEOUT = 10

    print("System active. Waiting for motion...")

    # 2. Main Loop
    while not unlocked:
        ret, frame = cap.read()
        if not ret:
            break

        # Check for motion first to save power
        if detect_motion(frame1, frame2):
            locations, names = recognize_faces(frame, known_encs, known_names)
            last_motion_time = time.time()

            if names and "Aymen" in names:
                print("Aymen detected! Unlocking...")
                os.system(GREET_COMMAND)
                os.system(UNLOCK_COMMAND)
                unlocked = True
            if not snapshot_taken and "Unknown" in names:
                timestamp = int(time.time())
                filename = f"unknown_{timestamp}.jpg"
                filepath = os.path.join(SNAPSHOT_DIR, filename)
                cv2.imwrite(filepath, frame)
                print(f"[!] Snapshot saved: {filename}")
                snapshot_taken = True
        else:
            if snapshot_taken and last_motion_time:
                if time.time() - last_motion_time > MOTION_TIMEOUT:
                    snapshot_taken = False
                    last_motion_time = None
                    print("[*] Motion timeout reached, flag reset.")
        
        # Prepare for next iteration
        frame1 = frame2
        frame2 = frame.copy()

        # Optional: Exit if 'q' is pressed (if window is active)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # 3. Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()