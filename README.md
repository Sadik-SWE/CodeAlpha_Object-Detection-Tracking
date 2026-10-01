---
title: Real-Time Object Detection and Tracking
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.1.0"
app_file: app.py
pinned: false
---

# 🎯 Real-Time Object Detection and Tracking

A real-time multi-object detection and tracking application built using Python, YOLO11, ByteTrack, OpenCV, and Gradio.

## ✨ Features

- 📷 Real-time webcam input
- 🔍 Multi-object detection
- 📦 Bounding boxes
- 🏷️ Object labels
- 🆔 Tracking IDs
- 🔄 ByteTrack object tracking
- 📊 Detected object counting
- 🌐 Browser-based interface

## 🧠 Technologies

- Python
- YOLO11
- Ultralytics
- ByteTrack
- OpenCV
- Gradio

## ⚙️ How It Works

```text
Webcam
   ↓
Gradio
   ↓
OpenCV
   ↓
YOLO11 Object Detection
   ↓
ByteTrack
   ↓
Bounding Boxes + Labels + Tracking IDs
   ↓
Real-Time Output