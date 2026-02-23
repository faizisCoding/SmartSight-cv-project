import cv2
import numpy as np
import time
import os
from collections import OrderedDict


scalefactor = 0.3
linepoints = []
frametodrawon = None


def select_line_callback(event, x, y, flags, param):
    global linepoints, frametodrawon
    if event == cv2.EVENT_LBUTTONDOWN and len(linepoints) < 2:
        originalx = int(x / scalefactor)
        originaly = int(y / scalefactor)
        linepoints.append((originalx, originaly))
        cv2.circle(frametodrawon, (x, y), 5, (0, 0, 255), -1)
        if len(linepoints) == 2:
            p1scaled = (int(linepoints[0][0] * scalefactor), int(linepoints[0][1] * scalefactor))
            p2scaled = (int(linepoints[1][0] * scalefactor), int(linepoints[1][1] * scalefactor))
            cv2.line(frametodrawon, p1scaled, p2scaled, (0, 255, 255), 2)
            cv2.imshow("Select Line", frametodrawon)


def select_line_first_frame(firstframe):
    global linepoints, frametodrawon
    linepoints = []
    frametodrawon = cv2.resize(firstframe, None, fx=scalefactor, fy=scalefactor)
    cv2.namedWindow("Select Line")
    cv2.setMouseCallback("Select Line", select_line_callback)
    print("Select two points as violation (red light/stop) line. Press 'c' to confirm, 'r' to reset, 'q' to quit.")
    while True:
        cv2.imshow("Select Line", frametodrawon)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if len(linepoints) == 2:
                cv2.destroyWindow("Select Line")
                return linepoints
            else:
                print("Select exactly two points.")
        elif key == ord('r'):
            print("Resetting points.")
            linepoints = []
            frametodrawon = cv2.resize(firstframe, None, fx=scalefactor, fy=scalefactor)
        elif key == ord('q'):
            print("Exiting selection.")
            cv2.destroyAllWindows()
            return None


def get_line_side(point, line_start, line_end):
    return (point[0] - line_start[0])*(line_end[1] - line_start[1]) - (point[1] - line_start[1])*(line_end[0] - line_start[0])


def estimate_speed(positions, px_to_meter=0.05, fps=30, min_frames=10, min_pixels=40):
    # stricter requirements for reliable speed
    if len(positions) < min_frames:
        return 0
    f1, pt1 = positions[0]
    f2, pt2 = positions[-1]
    pixeldist = np.linalg.norm(np.array(pt2)-np.array(pt1))
    if pixeldist < min_pixels:
        return 0
    meters = pixeldist * px_to_meter
    time_sec = (f2 - f1) / fps
    if time_sec == 0:
        return 0
    speed_mps = meters / time_sec
    speed_kmh = speed_mps * 3.6
    return speed_kmh


def detect_vehicles(frame, bgsubtractor, min_area=500):
    fgmask = bgsubtractor.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vehicles = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            centroid = (int(x+w/2), int(y+h/2))
            vehicles.append((centroid, x, y, w, h))
    return vehicles


def save_violation_frame(frame, frame_number, save_dir=r"{save_location}"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = f"violation_{frame_number}.jpg"
    cv2.imwrite(os.path.join(save_dir, filename), frame)


def main(video_path, output_path="output.mp4", speed_limit_kmh=80, px_to_meter=0.05):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video.")
        return
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read video.")
        return


    violation_line = select_line_first_frame(first_frame)
    if violation_line is None:
        cap.release()
        return
    p1, p2 = violation_line


    bgsubtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    vehicles = OrderedDict()
    vehicle_id_counter = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))


    frame_number = 0
    violation_memory = {}
    overspeeded_vehicles = set()
    redlight_violated_vehicles = set()

    # Create window with WINDOW_NORMAL to allow resizing
    cv2.namedWindow("Traffic Violation Detection", cv2.WINDOW_NORMAL)


    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1


        detected_vehicles = detect_vehicles(frame, bgsubtractor)
        updated_vehicles = {}


        for centroid, x, y, w, h in detected_vehicles:
            min_dist = float('inf')
            matched_id = None
            for vid, data in vehicles.items():
                prev_centroid = data["centroid"]
                dist = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))
                if dist < 50 and dist < min_dist:
                    min_dist = dist
                    matched_id = vid
            if matched_id is not None:
                positions = vehicles[matched_id]["positions"]
                positions.append((frame_number, centroid))
                updated_vehicles[matched_id] = {"centroid": centroid, "last_frame": frame_number, "positions": positions}
            else:
                vehicle_id_counter += 1
                updated_vehicles[vehicle_id_counter] = {"centroid": centroid, "last_frame": frame_number, "positions": [(frame_number, centroid)]}


        lost_ids = [vid for vid, data in vehicles.items() if frame_number - data["last_frame"] > 10]
        for vid in lost_ids:
            speed = estimate_speed(vehicles[vid]["positions"], px_to_meter, fps)
            positions = vehicles[vid]["positions"]
            if len(positions) >= 2:
                side_first = get_line_side(positions[0][1], p1, p2)
                side_last = get_line_side(positions[-1][1], p1, p2)
                if side_first * side_last < 0 and vid not in redlight_violated_vehicles:
                    violation_memory[vid] = frame_number
                    redlight_violated_vehicles.add(vid)
            if speed > speed_limit_kmh and vid not in overspeeded_vehicles:
                overspeeded_vehicles.add(vid)
            vehicles.pop(vid)
        vehicles = updated_vehicles


        for vid, data in vehicles.items():
            c = data["centroid"]
            x, y = c
            # stricter speed estimation here
            speed = estimate_speed(data["positions"], px_to_meter, fps, min_frames=10, min_pixels=40)
            color = (0, 0, 255) if speed > speed_limit_kmh else (0, 255, 0)
            violation_texts = []
            is_violation = False
            positions = data["positions"]


            # Overspeeding only once if valid
            if speed > speed_limit_kmh and vid not in overspeeded_vehicles:
                cv2.rectangle(frame, (x-25, y-25), (x+25, y+25), (0,0,255), 3)
                cv2.putText(frame, "OVERSPEEDING!", (x-20, y-35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 4)
                cv2.putText(frame, f"ID{vid} {round(speed,1)}km/h", (x-20, y-55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                violation_texts.append("OVERSPEEDING")
                is_violation = True
                overspeeded_vehicles.add(vid)


            # Red light jump detection only once
            if len(positions) >= 2:
                prev_point = positions[-2][1]
                curr_point = positions[-1][1]
                side_prev = get_line_side(prev_point, p1, p2)
                side_curr = get_line_side(curr_point, p1, p2)
                if (side_prev * side_curr < 0 or (vid in violation_memory and frame_number - violation_memory[vid] <= 30)) and vid not in redlight_violated_vehicles:
                    cv2.rectangle(frame, (x-30, y-30), (x+30, y+30), (0, 0, 255), 4)
                    cv2.putText(frame, "!!! RED LIGHT JUMP !!!", (frame.shape[1] // 4, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5)
                    violation_texts.append("RED LIGHT JUMP")
                    is_violation = True
                    violation_memory[vid] = frame_number
                    redlight_violated_vehicles.add(vid)


            if is_violation:
                save_violation_frame(frame, frame_number)


            if not is_violation and speed <= speed_limit_kmh:
                cv2.rectangle(frame, (x-20, y-20), (x+20, y+20), color, 2)
                cv2.putText(frame, f"ID{vid} {round(speed,1)}km/h", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


            if violation_texts:
                text = " & ".join(violation_texts)
                cv2.putText(frame, text, (x, y-95), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)


        cv2.line(frame, p1, p2, (0,255,255), 2)
        cv2.putText(frame, "Violation Line", (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)


        out.write(frame)
        cv2.imshow("Traffic Violation Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Annotated video saved as {output_path}")


if __name__ == "__main__":
    main(
        "{input path}",
        output_path="{output path}"
    )
