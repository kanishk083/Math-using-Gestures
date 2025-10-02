# Math-using-Gestures
Math-Gesture-Solver is a real-time, AI-powered system that allows users to solve mathematical problems by drawing them in the air using hand gestures. It combines computer vision for hand tracking (cvzone/OpenCV) with a dynamic web interface and an advanced Generative AI model, for equation recognition and solution generation.

🖐️ Math-Gesture-Solver: AI-Powered Math with Hand Gestures
Solve math problems in real-time by drawing them on your screen using natural hand movements! This project demonstrates a powerful fusion of computer vision and generative AI to create an interactive, hands-free educational tool.

✨ Features
Real-Time Hand Tracking: Utilizes cvzone's HandDetector (based on MediaPipe) for fast and accurate hand landmark detection.

Gesture-Based Drawing: Recognizes a specific hand gesture (e.g., index finger up) to enable "drawing" on a virtual canvas overlaid on the webcam feed.

Dynamic UI with Streamlit: Provides a clean, interactive web interface to display the live video feed, the drawing canvas, and the AI-generated solution.

AI-Powered Solving: When a "submit" gesture is detected (e.g., all fingers down except the thumb), the drawn content is sent to the Gemini API for interpretation and instant solution.

Core CV Libraries: Built on top of OpenCV (cv2) for video stream processing and image manipulation.
