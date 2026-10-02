import os
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

    try:
        # Gradio gives RGB image
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

        # Draw bounding boxes, labels and tracking IDs
        annotated_frame = result.plot()

        detected_objects = []

        # ==========================================
        # Get detected objects
        # ==========================================

        if result.boxes is not None:

            for box in result.boxes:

                # Class ID
                class_id = int(box.cls[0])

                # Object name
                object_name = model.names[class_id]

                # Tracking ID
                if box.id is not None:
                    track_id = int(box.id[0])
                    label = f"{object_name} ID:{track_id}"
                else:
                    label = object_name

                detected_objects.append(label)

        # ==========================================
        # Count objects
        # ==========================================

        object_names = []

        for item in detected_objects:
            object_names.append(item.split(" ID:")[0])

        counts = Counter(object_names)

        # ==========================================
        # Information text
        # ==========================================

        if counts:

            info = "Detected Objects:\n\n"

            for name, count in counts.items():
                info += f"{name}: {count}\n"

            info += "\nTracking IDs:\n\n"

            for item in detected_objects:
                info += f"{item}\n"

        else:

            info = "No supported objects detected."

        # Convert BGR back to RGB
        annotated_frame = cv2.cvtColor(
            annotated_frame,
            cv2.COLOR_BGR2RGB
        )

        return annotated_frame, info

    except Exception as e:

        return None, f"Error: {str(e)}"


# ==========================================
# Gradio Interface
# ==========================================

with gr.Blocks(
    title="Real-Time Object Detection and Tracking"
) as demo:

    gr.Markdown(
        """
        # 🎯 Real-Time Object Detection & Tracking

        ### YOLO11 + ByteTrack + OpenCV

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

    object_info = gr.Textbox(
        label="📊 Detection Information",
        lines=10,
        interactive=False
    )

    # Real-time webcam processing
    webcam.stream(
        fn=detect_and_track,
        inputs=webcam,
        outputs=[output, object_info],
        stream_every=0.20,
        concurrency_limit=1
    )


# ==========================================
# Launch
# ==========================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 7860))

    demo.queue()

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True
    )