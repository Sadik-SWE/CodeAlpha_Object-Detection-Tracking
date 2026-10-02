# 🎯 Real-Time Object Detection and Tracking

A real-time computer vision application that detects, identifies, and tracks multiple objects from a webcam using **YOLO11**, **OpenCV**, **NumPy**, and **Gradio**.

The application provides a browser-based interface where users can access their webcam, detect objects in real time, view bounding boxes, count detected objects, and monitor unique tracking IDs.

---

## 📌 Project Overview

Real-Time Object Detection and Tracking is a computer vision project developed as part of the **CodeAlpha Internship**.

The system processes live webcam frames and performs object detection using the **YOLO11n** model. A lightweight custom tracking algorithm then associates detected objects across consecutive frames and assigns unique tracking IDs.

### Processing Pipeline

```text
Webcam
   │
   ▼
Gradio Webcam Interface
   │
   ▼
OpenCV Frame Processing
   │
   ▼
YOLO11 Object Detection
   │
   ▼
Object Coordinates & Classes
   │
   ▼
Custom Object Tracking
   │
   ▼
Tracking IDs & Object Counts
   │
   ▼
Annotated Video Output
