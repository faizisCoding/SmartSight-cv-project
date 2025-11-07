# red_light_detector.py
#
# Description:
# This program detects vehicles that cross a user-defined line in a video stream,
# simulating red-light violation detection. It uses classical computer vision
# techniques available in OpenCV, without relying on any deep learning models.
#
# How to Run:
# 1. Ensure you have Python 3, OpenCV, and NumPy installed:
#    pip install opencv-python numpy
#
# 2. Update the `video_path` variable inside the `main()` function to point to your video file.
#
# 3. Run the script from your terminal:
#    python red_light_detector.py
#
# 4. Interactive Line Selection:
#    - The first frame of the video will be displayed.
#    - Click on two points on the frame to draw the "footpath" or stop line.
#    - Press 'c' to confirm the line, 'r' to reset, or 'q' to quit.


import cv2
import numpy as np
import os
from collections import OrderedDict


# --- Global variables for mouse callback ---
line_points = []
frame_to_draw_on = None
scale_factor = 0.3

def select_line_callback(event, x, y, flags, param):
    """
    Mouse callback function to select two points for the violation line.
    It now handles scaling from the resized window to the original frame.
    """
    global line_points, frame_to_draw_on

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(line_points) < 2:
            # Scale the clicked point back to the original frame size
            original_x = int(x / scale_factor)
            original_y = int(y / scale_factor)
            line_points.append((original_x, original_y))

            # Draw feedback on the displayed (resized) frame
            cv2.circle(frame_to_draw_on, (x, y), 5, (0, 0, 255), -1)
            
            if len(line_points) == 2:
                # Also draw the line on the feedback window
                p1_scaled = (int(line_points[0][0] * scale_factor), int(line_points[0][1] * scale_factor))
                p2_scaled = (int(line_points[1][0] * scale_factor), int(line_points[1][1] * scale_factor))
                cv2.line(frame_to_draw_on, p1_scaled, p2_scaled, (0, 255, 255), 2)
            
            cv2.imshow("Select Line", frame_to_draw_on)

def select_line(first_frame):
    """
    Displays a resized first frame to let the user draw a line.
    """
    global line_points, frame_to_draw_on
    line_points = []
    
    # Create a resized copy for display
    frame_to_draw_on = cv2.resize(first_frame, (0, 0), fx=scale_factor, fy=scale_factor)

    cv2.namedWindow("Select Line")
    cv2.setMouseCallback("Select Line", select_line_callback)
    
    print("Please select two points on the image to define the footpath line.")
    print("Press 'c' to confirm the line, 'r' to reset, or 'q' to quit.")

    while True:
        cv2.imshow("Select Line", frame_to_draw_on)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            if len(line_points) == 2:
                cv2.destroyWindow("Select Line")
                return line_points
            else:
                print("Please select exactly two points before confirming.")
        
        elif key == ord('r'):
            print("Resetting points. Please select again.")
            line_points = []
            # Also reset the visual feedback frame
            frame_to_draw_on = cv2.resize(first_frame, (0, 0), fx=scale_factor, fy=scale_factor)

        elif key == ord('q'):
            print("Quitting.")
            cv2.destroyAllWindows()
            return None

def detect_objects(frame, bg_subtractor, min_area):
    """
    Detects moving objects using background subtraction and contour filtering.
    """
    fg_mask = bg_subtractor.apply(frame)
    
    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((8, 8), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)
    
    _, thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            
    return valid_contours

def get_line_side(p, line_p1, line_p2):
    """
    Determines which side of a line a point is on.
    """
    return (p[0] - line_p1[0]) * (line_p2[1] - line_p1[1]) - (p[1] - line_p1[1]) * (line_p2[0] - line_p1[0])

def save_violation_screenshot(frame, output_dir, object_id, frame_no):
    """
    Saves a screenshot of the frame where a violation occurred.
    The filename includes the object ID and the frame number to be unique.

    Args:
        frame (np.ndarray): The video frame to save.
        output_dir (str): The directory to save the screenshot in.
        object_id (int): The ID of the violating object.
        frame_no (int): The current frame number.
    """
    # Create a unique filename for the screenshot
    filename = f"violation_ID-{object_id}_frame-{frame_no}.jpg"
    filepath = os.path.join(output_dir, filename)

    # Save the frame as a JPG image
    try:
        cv2.imwrite(filepath, frame)
        print(f"    -> Violation detected! Screenshot saved: {os.path.basename(filepath)}")
    except Exception as e:
        print(f"    -> Error saving screenshot for ID {object_id}: {e}")

def update_tracks(tracked_objects, detected_centroids, max_dist, disappear_frames, speed_thresh, next_id):
    """
    Updates object tracks based on centroid matching.
    """
    if not detected_centroids:
        for obj_id in list(tracked_objects.keys()):
            tracked_objects[obj_id]["disappeared"] += 1
            if tracked_objects[obj_id]["disappeared"] > disappear_frames:
                del tracked_objects[obj_id]
        return tracked_objects, next_id

    if not tracked_objects:
        for centroid in detected_centroids:
            tracked_objects[next_id] = {
                "centroid": centroid, "disappeared": 0,
                "last_centroid": centroid, "moving": False
            }
            next_id += 1
        return tracked_objects, next_id

    object_ids = list(tracked_objects.keys())
    object_centroids = [tracked_objects[oid]["centroid"] for oid in object_ids]
    
    dist_matrix = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - np.array(detected_centroids), axis=2)

    used_rows, used_cols = set(), set()
    rows = dist_matrix.min(axis=1).argsort()
    
    for row in rows:
        if row in used_rows: continue
        col = dist_matrix[row].argmin()
        if col in used_cols or dist_matrix[row, col] > max_dist: continue

        obj_id = object_ids[row]
        new_centroid = detected_centroids[col]
        movement_dist = np.linalg.norm(np.array(new_centroid) - np.array(tracked_objects[obj_id]["centroid"]))
        
        tracked_objects[obj_id]["last_centroid"] = tracked_objects[obj_id]["centroid"]
        tracked_objects[obj_id]["centroid"] = new_centroid
        tracked_objects[obj_id]["disappeared"] = 0
        tracked_objects[obj_id]["moving"] = movement_dist > speed_thresh

        used_rows.add(row)
        used_cols.add(col)

    unmatched_rows = set(range(dist_matrix.shape[0])) - used_rows
    for row in unmatched_rows:
        obj_id = object_ids[row]
        tracked_objects[obj_id]["disappeared"] += 1
        if tracked_objects[obj_id]["disappeared"] > disappear_frames:
            del tracked_objects[obj_id]

    unmatched_cols = set(range(dist_matrix.shape[1])) - used_cols
    for col in unmatched_cols:
        centroid = detected_centroids[col]
        tracked_objects[next_id] = {
            "centroid": centroid, "disappeared": 0,
            "last_centroid": centroid, "moving": False
        }
        next_id += 1
    return tracked_objects, next_id

def main():
    # --- FIXED PARAMETERS (Replaces command-line args) ---
    min_area = 3500
    max_distance = 75
    disappear_frames = 15
    speed_threshold = 5.0
    
    # --- 1. Video and Line Setup ---
    # !!! IMPORTANT: UPDATE THIS PATH TO YOUR VIDEO FILE !!!
    video_path ="C:\\Users\\Faiz\\Desktop\\programs\\CV\\DataSets\\Red_lights\\Red_Light_Violation_05.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    footpath_line = select_line(first_frame)
    if footpath_line is None:
        cap.release()
        return
        
    p1, p2 = footpath_line

    # --- 2. Initializations ---
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    
    tracked_objects = OrderedDict()
    violation_ids = set()
    next_object_id = 0
    frame_no = 0  # Frame counter for unique filenames

    # Video Writer Setup
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define the specific output folder and create it
    output_folder = r"C:\Users\Faiz\Desktop\programs\CV\DataSets\Red_lights\Violations"
    os.makedirs(output_folder, exist_ok=True)
    
    # Set the full path for the output video file
    output_path = os.path.join(output_folder, 'output_annotated.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print("Processing video...")
    
    # --- 3. Main Processing Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1  # Increment frame counter

        # --- 4. Detection ---
        valid_contours = detect_objects(frame, bg_subtractor, min_area)
        
        detected_centroids = []
        contour_map = {}
        for i, c in enumerate(valid_contours):
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                detected_centroids.append((cx, cy))
                contour_map[i] = c
                
        # --- 5. Tracking ---
        tracked_objects, next_object_id = update_tracks(
            tracked_objects, detected_centroids,
            max_distance, disappear_frames, speed_threshold,
            next_object_id
        )

        # --- 6. Violation Check & Drawing ---
        new_violations_in_frame = [] # List to hold new violators in this frame
        for obj_id, data in tracked_objects.items():
            is_violation = obj_id in violation_ids
            
            # Check for a new violation (line cross)
            if data["moving"] and not is_violation:
                prev_side = get_line_side(data["last_centroid"], p1, p2)
                current_side = get_line_side(data["centroid"], p1, p2)
                if prev_side * current_side < 0:
                    is_violation = True
                    violation_ids.add(obj_id)
                    new_violations_in_frame.append(obj_id) # Mark this ID for a screenshot

            # Find the corresponding contour to draw a bounding box
            current_centroid = data["centroid"]
            min_dist = float('inf')
            matched_contour = None
            for i, centroid in enumerate(detected_centroids):
                dist = np.linalg.norm(np.array(current_centroid) - np.array(centroid))
                if dist < min_dist and dist < max_distance:
                    min_dist = dist
                    matched_contour = contour_map.get(i)
            
            # Draw bounding boxes and labels on the frame
            if matched_contour is not None:
                (x, y, w, h) = cv2.boundingRect(matched_contour)
                box_color = (0, 0, 255) if is_violation else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                
                text = f"ID: {obj_id}"
                cv2.putText(frame, text, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                if is_violation:
                    cv2.putText(frame, "VIOLATION", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw the violation line on the frame
        cv2.line(frame, p1, p2, (0, 255, 255), 2)

        # Save screenshots for any new violations detected in this frame
        for violator_id in new_violations_in_frame:
            save_violation_screenshot(frame, output_folder, violator_id, frame_no)
        
        # Write the annotated frame to the output video
        out.write(frame)
        
        # Display the output
        screen_scale = 0.6
        frame_display = cv2.resize(frame, (0, 0), fx=screen_scale, fy=screen_scale)
        cv2.imshow("Red Light Detection", frame_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # --- 7. Cleanup ---
    print(f"Processing complete. Annotated video saved to: {output_path}")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
