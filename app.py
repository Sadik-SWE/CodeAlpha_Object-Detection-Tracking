import cv2
import gradio as gr
from ultralytics import YOLO
from collections import Counter


# ==========================================
# Load YOLO Model
# ==========================================

model = YOLO("yolo11n.pt")


# ==========================================
# Object Detection + Tracking
# ==========================================

def detect_and_track(frame):

    if frame is None:
        return None, "No camera input"

    # Gradio gives RGB
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # YOLO detection + ByteTrack tracking
    results = model.track(
        source=frame_bgr,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.30,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    # Draw boxes, labels and tracking IDs
    annotated_frame = result.plot()

    # ==========================================
    # Get detected objects
    # ==========================================

    detected_objects = []

    if result.boxes is not None:

        for box in result.boxes:

            # Class ID
            class_id = int(box.cls[0])

            # Object name
            object_name = model.names[class_id]

            # Tracking ID
            if box.id is not None:
                track_id = int(box.id[0])
                label = f"{object_name} {track_id}"
            else:
                label = f"{object_name}"

            detected_objects.append(label)

    # ==========================================
    # Object Counter
    # ==========================================

    object_names = [
        item.split()[0]
        for item in detected_objects
    ]

    counts = Counter(object_names)

    # Create information text
    if counts:

        info = "### 🎯 Detected Objects\n\n"

        for name, count in counts.items():
            info += f"- **{name}**: {count}\n"

        info += "\n### 🆔 Tracking IDs\n\n"

        for item in detected_objects:
            info += f"- `{item}`\n"

    else:
        info = "### 🔍 No supported objects detected"

    # RGB for Gradio
    annotated_frame = cv2.cvtColor(
        annotated_frame,
        cv2.COLOR_BGR2RGB
    )

    return annotated_frame, info


# ==========================================
# Gradio UI
# ==========================================

with gr.Blocks(
    title="Object Detection and Tracking"
) as demo:

    gr.Markdown(
        """
        # 🎯 Real-Time Object Detection & Tracking

        ### YOLO + ByteTrack + OpenCV

        Detect and track multiple objects using your webcam.

        **Supported examples:**
        Person • Car • Bicycle • Motorcycle • Bus • Bottle •
        Laptop • Cell Phone • Chair • Dog • Cat • Backpack
        """
    )

    with gr.Row():

        # Webcam
        webcam = gr.Image(
            sources=["webcam"],
            type="numpy",
            streaming=True,
            label="📷 Webcam"
        )

        # Output
        output = gr.Image(
            type="numpy",
            label="🎯 Detection & Tracking"
        )

    # Object information
    object_info = gr.Markdown(
        "### 🔍 Waiting for camera..."
    )

    # Start real-time processing
    webcam.stream(
        fn=detect_and_track,
        inputs=webcam,
        outputs=[output, object_info],
        stream_every=0.15
    )


# ==========================================
# Start Application
# ==========================================

if __name__ == "__main__":
    import os

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )