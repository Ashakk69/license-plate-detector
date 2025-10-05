# 🚗 Vehicle Entry System - License Plate Detector

A real-time license plate detection and recognition system built with Python, YOLOv8, OpenCV, and Flask. This application can stream video from multiple cameras, detect license plates, and log them into a database.

![Project Demo GIF](link_to_your_gif.gif)  ---

## ✨ Features

-   **Multi-Camera Support:** Stream from webcams or IP cameras (via RTSP).
-   **High-Accuracy Detection:** Uses a YOLOv8 model for robust license plate detection.
-   **OCR Engine Options:** Supports both Tesseract and EasyOCR for character recognition.
-   **Universal Plate Filter:** A flexible filter to detect various international plate formats.
-   **Database Logging:** Automatically logs every detected plate with a timestamp.
-   **Web Interface:** A clean web UI built with Flask and Bootstrap to view live feeds and logs.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%235C3EE8.svg?style=for-the-badge&logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Bootstrap](https://img.shields.io/badge/bootstrap-%238511FA.svg?style=for-the-badge&logo=bootstrap&logoColor=white)

---

## 🚀 Getting Started

### Prerequisites

-   Python 3.8+
-   Git
-   Tesseract OCR Engine installed on your system

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Ashakk69/license-plate-detector.git](https://github.com/Ashakk69/license-plate-detector.git)
    cd license-plate-detector
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You'll need to create a `requirements.txt` file by running `pip freeze > requirements.txt` in your terminal)*

### Running the Application

1.  The YOLOv8 model will be downloaded automatically on the first run.

2.  Start the Flask server:
    ```bash
    python app.py
    ```

3.  Open your web browser and navigate to `http://127.0.0.1:5000`.

---

## 🖼️ Screenshots

| Live Feed                                   | Detection Logs                                |
| ------------------------------------------- | --------------------------------------------- |
| ![Live Feed Screenshot](link_to_screenshot_1.jpg) | ![Logs Screenshot](link_to_screenshot_2.jpg) |
---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
