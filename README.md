# 🧠 Object Detection Using OpenCV and Python

## 📌 Project Overview

This project focuses on implementing **Object Detection** using **OpenCV** and **Python**. The system is designed to detect and localize objects (e.g., vehicles, people) in images or video feeds by applying core image processing techniques like **thresholding, background subtraction, erosion, dilation, and contour detection**.

It demonstrates how basic computer vision methods can be used to create an efficient and lightweight object detection system, especially useful in areas like **traffic monitoring**, **surveillance**, and **automated inspection**.

---

## 🎯 Objectives

* Detect moving objects in video streams using Python and OpenCV.
* Implement preprocessing techniques to improve detection accuracy.
* Visualize object detection results in real-time.
* Understand and apply OpenCV functions such as `cv2.threshold()`, `cv2.erode()`, `cv2.dilate()`, and `cv2.findContours()`.

---

## 🛠️ Technologies Used

* **Python 3.11**
* **OpenCV (cv2)**
* **NumPy**

---

## 📷 Image Processing Techniques Used

* **Thresholding** – Segment image pixels for binary classification.
* **Background Subtraction** – Isolate moving objects from static background.
* **Erosion & Dilation** – Refine foreground masks.
* **Contour Detection** – Identify object boundaries for drawing bounding boxes.

---

## ⚙️ How It Works

### 1. Read the Video File

```python
capture = cv2.VideoCapture('ambulance.mkv')
```

### 2. Apply Background Subtraction

```python
backgroundObject = cv2.createBackgroundSubtractorMOG2()
fgmask = backgroundObject.apply(frame)
```

### 3. Preprocess the Mask

```python
fgmask = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)[1]
fgmask = cv2.erode(fgmask, kernel, iterations=1)
fgmask = cv2.dilate(fgmask, kernel2, iterations=4)
```

### 4. Detect and Label Objects

```python
for cnt in contours:
    if cv2.contourArea(cnt) > 5000:
        x, y, width, height = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(frame, "OBJECT DETECTED", (x, y - 10), ...)
```

---

## 📈 Results

The system is capable of identifying and highlighting moving vehicles in a video using bounding boxes. It operates in near real-time with decent accuracy and speed for basic traffic monitoring applications.

---

## 📦 Installation & Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/object-detection-opencv.git
   cd object-detection-opencv
   ```

2. Install required libraries:

   ```bash
   pip install opencv-python numpy
   ```

3. Run the script:

   ```bash
   python object_detection.py
   ```

---

## 🚗 Applications

* **Traffic Surveillance**
* **People Counting**
* **Retail Inventory Management**
* **Security and Face Recognition**
* **Industrial Quality Checks**

---

## 📚 References

* [OpenCV Documentation](https://docs.opencv.org/)
* [GeeksforGeeks – OpenCV in Python](https://www.geeksforgeeks.org/opencv-python-tutorial/)
* [Object Detection Tutorials (YouTube)](https://youtu.be/oXlwWbU8l2o)
* Internship Report by Mansuri Mo. Umer, A.Y. Dadabhai Technical Institute (2024–25)

---

## 🧑‍💻 Author

**Mansuri Mo. Umer**
**Department of Computer Engineering**
**A.Y. Dadabhai Technical Institute**
**Enrollment No: 226010307052**

Project Guide: **Ms. Anjali V. Patel**
