import os
import cv2
import gradio as gr
import numpy as np

from ultralytics import YOLO


# =========================================================
# YOLO MODEL
# =========================================================

model = YOLO("yolo11n.pt")

# Force CPU for Render
model.to("cpu")


# =========================================================
# SIMPLE OBJECT TRACKER
# =========================================================

class SimpleTracker:

    def __init__(self, max_distance=80, max_missing=10):

        self.next_id = 1

        self.tracks = {}

        self.max_distance = max_distance
        self.max_missing = max_missing


    def calculate_distance(self, point1, point2):

        return np.sqrt(
            (point1[0] - point2[0]) ** 2
            +
            (point1[1] - point2[1]) ** 2
        )


    def update(self, detections):

        """
        detections:

        [
            {
                "class_id": 0,
                "name": "person",
                "center": (x, y),
                "box": (x1, y1, x2, y2)
            }
        ]
        """

        updated_tracks = {}

        used_track_ids = set()


        # =====================================================
        # Match new detections with existing tracks
        # =====================================================

        for detection in detections:

            best_id = None
            best_distance = self.max_distance

            for track_id, track in self.tracks.items():

                if track_id in used_track_ids:
                    continue

                # Same object class only
                if track["class_id"] != detection["class_id"]:
                    continue

                distance = self.calculate_distance(
                    track["center"],
                    detection["center"]
                )

                if distance < best_distance:

                    best_distance = distance
                    best_id = track_id


            # =================================================
            # Existing track found
            # =================================================

            if best_id is not None:

                track_id = best_id

                used_track_ids.add(track_id)

                updated_tracks[track_id] = {

                    "class_id": detection["class_id"],

                    "name": detection["name"],

                    "center": detection["center"],

                    "box": detection["box"],

                    "missing": 0
                }


            # =================================================
            # New object
            # =================================================

            else:

                track_id = self.next_id

                self.next_id += 1

                updated_tracks[track_id] = {

                    "class_id": detection["class_id"],

                    "name": detection["name"],

                    "center": detection["center"],

                    "box": detection["box"],

                    "missing": 0
                }


        # =====================================================
        # Keep temporarily missing tracks
        # =====================================================

        for track_id, track in self.tracks.items():

            if track_id not in updated_tracks:

                track["missing"] += 1

                if track["missing"] <= self.max_missing:

                    updated_tracks[track_id] = track


        self.tracks = updated_tracks

        return self.tracks


# =========================================================
# CREATE TRACKER
# =========================================================

tracker = SimpleTracker(
    max_distance=100,
    max_missing=8
)


# =========================================================
# OBJECT DETECTION + TRACKING
# =========================================================

def detect_and_track(frame):

    if frame is None:

        return None, "### 🔍 Waiting for webcam..."


    try:

        # =================================================
        # Gradio gives RGB
        # =================================================

        frame_bgr = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )


        # =================================================
        # YOLO DETECTION
        # =================================================

        results = model.predict(

            source=frame_bgr,

            conf=0.25,

            imgsz=320,

            device="cpu",

            verbose=False
        )


        result = results[0]


        detections = []


        # =================================================
        # Extract detections
        # =================================================

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls[0])

                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )


                center_x = int(
                    (x1 + x2) / 2
                )

                center_y = int(
                    (y1 + y2) / 2
                )


                object_name = model.names[class_id]


                detections.append({

                    "class_id": class_id,

                    "name": object_name,

                    "center": (
                        center_x,
                        center_y
                    ),

                    "box": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),

                    "confidence": confidence
                })


        # =================================================
        # TRACK OBJECTS
        # =================================================

        tracks = tracker.update(
            detections
        )


        # =================================================
        # Draw tracking
        # =================================================

        output_frame = frame_bgr.copy()


        for track_id, track in tracks.items():

            # Ignore temporarily missing objects
            if track["missing"] > 0:

                continue


            x1, y1, x2, y2 = track["box"]

            name = track["name"]

            center_x, center_y = track["center"]


            # -------------------------------------------------
            # Bounding box
            # -------------------------------------------------

            cv2.rectangle(

                output_frame,

                (x1, y1),

                (x2, y2),

                (0, 255, 0),

                2
            )


            # -------------------------------------------------
            # Center point
            # -------------------------------------------------

            cv2.circle(

                output_frame,

                (center_x, center_y),

                5,

                (0, 0, 255),

                -1
            )


            # -------------------------------------------------
            # Label
            # -------------------------------------------------

            label = f"{name} | ID:{track_id}"


            cv2.rectangle(

                output_frame,

                (x1, max(0, y1 - 30)),

                (
                    x1 + len(label) * 10 + 10,
                    y1
                ),

                (0, 255, 0),

                -1
            )


            cv2.putText(

                output_frame,

                label,

                (x1 + 5, y1 - 8),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (0, 0, 0),

                2
            )


        # =================================================
        # Count objects
        # =================================================

        object_names = []

        active_tracks = []


        for track_id, track in tracks.items():

            if track["missing"] == 0:

                object_names.append(
                    track["name"]
                )

                active_tracks.append(
                    f"{track['name']} ID:{track_id}"
                )


        from collections import Counter

        counts = Counter(
            object_names
        )


        # =================================================
        # Information panel
        # =================================================

        if counts:

            info = (
                "### 🎯 Detected Objects\n\n"
            )


            for name, count in counts.items():

                info += (
                    f"- **{name}**: {count}\n"
                )


            info += (
                "\n### 🆔 Tracking IDs\n\n"
            )


            for item in active_tracks:

                info += (
                    f"- `{item}`\n"
                )


        else:

            info = (
                "🔍 No objects detected."
            )


        # =================================================
        # Convert BGR → RGB
        # =================================================

        output_frame = cv2.cvtColor(

            output_frame,

            cv2.COLOR_BGR2RGB
        )


        return output_frame, info


    except Exception as e:

        print(
            "Detection error:",
            str(e)
        )


        return (

            frame,

            f"❌ Detection Error: `{str(e)}`"
        )


# =========================================================
# GRADIO UI
# =========================================================

with gr.Blocks(

    title="Real-Time Object Detection and Tracking"

) as demo:


    gr.Markdown(

        """
        # 🎯 Real-Time Object Detection & Tracking

        ### YOLO11 + OpenCV + Custom Object Tracking

        Detect and track multiple objects using your webcam.

        **Examples:** Person • Car • Bicycle • Motorcycle • Bus •
        Bottle • Laptop • Cell Phone • Chair • Dog • Cat • Backpack
        """
    )


    with gr.Row():


        webcam = gr.Image(

            sources=["webcam"],

            type="numpy",

            streaming=True,

            label="📷 Webcam"
        )


        output = gr.Image(

            type="numpy",

            label="🎯 Detection & Tracking"
        )


    object_info = gr.Markdown(

        "### 🔍 Waiting for camera..."
    )


    webcam.stream(

        fn=detect_and_track,

        inputs=webcam,

        outputs=[
            output,
            object_info
        ],

        stream_every=0.5,

        concurrency_limit=1
    )


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    port = int(

        os.environ.get(
            "PORT",
            7860
        )
    )


    demo.queue()


    demo.launch(

        server_name="0.0.0.0",

        server_port=port,

        show_error=True
    )