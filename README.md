# RGB Overlay Tool

## 🔹 Overview
This Streamlit-based tool allows users to overlay an RGB image on top of a base satellite or spatial image. It is helpful for visual comparison and data blending in remote sensing workflows.

## 🔹 Features
- Upload base and RGB overlay images
- Adjust opacity to control blending strength
- Real-time preview of the combined image
- Downloadable output image

## 🔹 Requirements
```bash
pip install streamlit opencv-python pillow numpy

streamlit run rgb_overlay_app.py
