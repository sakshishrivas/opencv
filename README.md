# opencv   
Open cv is a Python open -source library which is used for computer vision in AI,ML ,FACE RECCOGNITION etc.
It helps in tasks like object detection, face recognition, image transformation, and more.
Includes several hundreds ofcomputer vision algorithms.
It has C++,Python,Java and Matplotlib interfaces and support windows,linux,Android and MacOS.
Opencv is wriiten natively in C++.
Initial release June 2000: 20 year ago
Open cv is available free of cost.
Since opencv library is written in C++/C ,so it is quite fast,It can be used with Python
Requires less RAM to usage ,it may be of 60-70 MB.
Computer vision is portable as opencv and can run on any device that can run on C.
It provides tools to analyze, modify, and process images and videos efficiently.
OpenCV is widely used in real-time applications like face recognition, object detection, motion tracking, and more.



Why Use OpenCV?
Open Source & Free – Available for everyone without cost.
Fast & Efficient – Optimized for real-time applications.
Supports Multiple Languages – Python, C++, Java, and more.
Cross-Platform – Works on Windows, Linux, macOS, and even mobile devices.
Extensive Community – A large community supports and maintains OpenCV.



Installation________________>>>>>>
(pip install opencv-python)

Key Features of OpenCV
Image Processing – Resize, crop, filter, and enhance images.
Object Detection – Identify objects like faces, cars, and people.
Feature Detection – Detect edges, contours, and key points in images.
Video Processing – Read, write, and modify video files in real time.
Machine Learning Integration – Works with AI models for recognition tasks.
Augmented Reality (AR) – Helps create AR applications like Snapchat filters.


Open cv task:--
object classification
object identification
object tracking
image resolution
feature matching
video motion analysis



Module Name	Description
cv2	Main OpenCV module for image & video processing (Python wrapper for OpenCV)
opencv_core	Core functionalities like matrix operations, data structures, and algorithms
opencv_imgproc	Image processing functions like filtering, transformations, and edge detection
opencv_videoio	Handles reading and writing video files and streams
opencv_objdetect	Object detection (face, eyes, car, etc.) using pre-trained classifiers
opencv_ml	Machine learning algorithms like SVM, KNN, and decision trees
opencv_dnn	Deep learning module (supports TensorFlow, Caffe, YOLO, etc.)
opencv_calib3d	Camera calibration, stereo vision, and 3D reconstruction
opencv_features2d	Feature detection (ORB, SIFT, SURF, FAST)
opencv_video	Motion tracking, optical flow, and background subtraction
opencv_photo	Image enhancement techniques (denoising, inpainting)
opencv_stitching	Image stitching (panorama creation)
opencv_highgui	GUI functions for displaying images and videos
opencv_imgcodecs	Encoding and decoding image formats (JPEG, PNG, BMP)

OpenCV is a powerful tool with different modules for:
✔ Image Processing (cv2.imgproc)
✔ Video Processing (cv2.videoio)
✔ Object Detection (cv2.objdetect)
✔ Machine Learning (cv2.ml)
✔ Deep Learning (cv2.dnn)



Basic OpenCV Operations
OpenCV is widely used for image and video processing. Here are some fundamental operations in OpenCV with Python.

📌 1. Read & Display an Image
Load an image and display it using OpenCV.

import cv2  

# Load the image
img = cv2.imread("image.jpg")  

# Display the image
cv2.imshow("Image Window", img)  

# Wait for a key press and close the window
cv2.waitKey(0)  
cv2.destroyAllWindows()
🎨 2. Convert Image to Grayscale
Convert a colored image into grayscale.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
✂ 3. Resize an Image
Change the dimensions of an image.
resized = cv2.resize(img, (300, 300))  # Resize to 300x300 pixels
cv2.imshow("Resized Image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
🔄 4. Rotate an Image
Rotate an image by 90 degrees.
rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
cv2.imshow("Rotated Image", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
✍ 5. Draw Shapes on an Image
Draw a rectangle, circle, and line on an image.
# Draw a rectangle
cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), 3)

# Draw a circle
cv2.circle(img, (250, 250), 50, (255, 0, 0), -1)  # -1 fills the shape

# Draw a line
cv2.line(img, (0, 0), (400, 400), (0, 0, 255), 2)

cv2.imshow("Shapes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
🔍 6. Detect Edges using Canny Edge Detection
Find edges in an image using the Canny algorithm.
edges = cv2.Canny(img, 100, 200)
cv2.imshow("Edge Detection", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
🖼 7. Blur an Image
Apply Gaussian blur to reduce noise in an image.
blurred = cv2.GaussianBlur(img, (15, 15), 0)
cv2.imshow("Blurred Image", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()
🤖 8. Face Detection using Haar Cascades
Detect faces in an image.
# Load pre-trained classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Draw rectangles around faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow("Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
🎥 9. Capture Video from Webcam
Read frames from the webcam and display them.
cap = cv2.VideoCapture(0)  # 0 for default webcam

while True:
    ret, frame = cap.read()
    cv2.imshow("Webcam Feed", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
🎭 10. Save an Image
Save an image after processing.
cv2.imwrite("output.jpg", img)  # Save image as output.jpg
























































































































