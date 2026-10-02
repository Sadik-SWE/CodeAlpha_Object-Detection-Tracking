import os
import cv2
import gradio as gr
from ultralytics import YOLO
from collections import Counter


# ==========================================
# Load YOLO Model
# ==========================================

model = YOLO("yolo11n.pt")

# Force CPU for Render
model.to("cpu")


# ==========================================
# Object Detection + Tracking
# ==========================================

def detect_and_track(frame):

    if frame is None:
        return None, "Waiting for webcam..."

    try:
        # Gradio gives RGB
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # YOLO + ByteTrack
        results = model.track(
            source=frame_bgr,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.25,
            imgsz=320,
            device="cpu",
            verbose=False
        )

        result = results[0]

        # Draw detections
        annotated_frame = result.plot()

        detected_objects = []

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls[0])
                object_name = model.names[class_id]

                if box.id is not None:
                    track_id = int(box.id[0])
                    detected_objects.append(
                        f"{object_name} ID:{track_id}"
                    )
                else:
                    detected_objects.append(
                        object_name
                    )

        # ==========================================
        # Count objects
        # ==========================================

        object_names = []

        for item in detected_objects:
            object_names.append(
                item.split(" ID:")[0]
            )

        counts = Counter(object_names)

        # ==========================================
        # Information
        # ==========================================

        if counts:

            info = "### 🎯 Detected Objects\n\n"

            for name, count in counts.items():
                info += f"- **{name}**: {count}\n"

            info += "\n### 🆔 Tracking IDs\n\n"

            for item in detected_objects:
                info += f"- `{item}`\n"

        else:

            info = "🔍 No objects detected in this frame."

        # BGR → RGB
        annotated_frame = cv2.cvtColor(
            annotated_frame,
            cv2.COLOR_BGR2RGB
        )

        return annotated_frame, info

    except Exception as e:

        # Return original frame instead of blank output
        return frame, f"❌ Detection Error: {str(e)}"


# ==========================================
# Gradio UI
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

    object_info = gr.Markdown(
        "### 🔍 Waiting for camera..."
    )

    webcam.stream(
        fn=detect_and_track,
        inputs=webcam,
        outputs=[output, object_info],
        stream_every=0.5,
        concurrency_limit=1
    )


# ==========================================
# Start Server
# ==========================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 7860)
    )

    demo.queue()

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True
    )