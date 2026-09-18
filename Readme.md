# 🔎 YOLOv11 Image Search

A Streamlit-based image search application powered by **YOLOv11 object detection**.

The application allows users to upload images, run YOLOv11 inference, extract detected objects and their metadata, and then search the uploaded images based on detected object classes and object counts.

Instead of manually checking every image, users can search for images containing specific objects using **OR / AND conditions** and optional count limits.

---

## Features

- Upload multiple images through the Streamlit interface
- YOLOv11 object detection
- Confidence-based detection
- Bounding box extraction
- Object counting
- Search images by detected object classes
- OR and AND search modes
- Maximum object-count filtering
- Display detected objects with bounding boxes
- View detection metadata
- Export detection results as JSON
- Streamlit-based interactive UI
- Deployable on Streamlit Community Cloud

---

## How It Works

The application follows a simple pipeline:

```text
Upload Images
      │
      ▼
YOLOv11 Inference
      │
      ▼
Object Detection
      │
      ├── Class
      ├── Confidence
      ├── Bounding Box
      └── Object Count
      │
      ▼
Detection Metadata
      │
      ▼
Search & Filtering
      │
      ├── OR Search
      ├── AND Search
      └── Count Filtering
      │
      ▼
Matching Images
```

## Live Demo

Try the deployed application here:

👉 **[YOLOv11 Image Search - Live Demo](https://yolov11imagesearch-2-h443a4vjbrjmfclnvxbvqi.streamlit.app/)**

Upload your images, run YOLOv11 object detection, and search the detected objects using **OR / AND search conditions** and **object-count filters** directly from the browser.