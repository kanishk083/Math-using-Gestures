import cvzone
import cv2
import numpy as np

from cvzone.HandTrackingModule import HandDetector
import google.genai as genai
from google.genai import types
from fontTools.misc.cython import returns
from PIL import Image
import streamlit as st


st.image("Navy Geometric Technology LinkedIn Banner.png")

col1, col2 = st.columns([1, 1])
with col1:
    run = st.checkbox("Run",value=True)
    FRAME_WINDOW=st.image([])
with col2:
    output_text_area = st.title("Answer")
    output_text_area = st.subheader("")

#api integration
api_key_manual = "AIzaSyBdRuCxVvQDLIICZz5BGMUJu9m4FG-VOV4"

client = genai.Client(api_key=api_key_manual)
gemini_model = "gemini-2.5-flash"


response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how AI works in a few words",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
    ),
)
print(response.text)


# Initialize the webcam to capture video
# The '2' indicates the third camera connected to your computer; '0' would usually refer to the built-in camera
cap = cv2.VideoCapture(0)
cap.set(propId=3,value=1280)
cap.set(propId=4,value=720)

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.8)

def getHandInfo(img):
# Find hands in the current frame
    # The 'draw' parameter draws landmarks and hand outlines on the image if set to True
    # The 'flipType' parameter flips the image, making it easier for some detections
    hands, img = detector.findHands(img, draw=True, flipType=True)

    # Check if any hands are detected
    if hands:
        # Information for the first hand detected
        hand = hands[0]  # Get the first hand detected
        lmList = hand["lmList"]  # List of 21 landmarks for the first hand


        # Count the number of fingers up for the first hand
        fingers = detector.fingersUp(hand)
        print(fingers)  # Print the count of fingers that are up
        return fingers , lmList
    else:
        return None

def draw(info,prev_pos, canvas):

  fingers,lmlist=info
  current_pos=None

  if fingers ==[0, 1, 0, 0 ,0]:
      current_pos = tuple(map(int, lmlist[8][0:2]))
      if prev_pos is None: prev_pos = current_pos
      cv2.line(canvas, current_pos, prev_pos, color=(255, 0, 0), thickness=10)
  elif fingers == [1, 0, 0, 0, 0]:
      canvas = np.zeros_like(img)


  return current_pos,canvas


def sendToAI(gemini_model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:
        pil_img = Image.fromarray(canvas)
        content_parts = [
            "Solve this math problem based on the image.",  # Your text prompt
            pil_img  # The image object
        ]
        response = client.models.generate_content(
            model=gemini_model,  # Use the passed-in model name
            contents=content_parts
        )
        return response.text
    return ""  # or "Waiting for gesture..."

prev_pos = None
canvas = None
img_combines = None

output = ""
#Continuously get frames from the webcam
while True:
    # Capture each frame from the webcam
    # 'success' will be True if the frame is successfully captured, 'img' will contain the frame
    success, img = cap.read()
    img = cv2.flip(img, 1)
    info = getHandInfo(img)

    if canvas is None:
        canvas=np.zeros_like(img)

    if info:
        fingers , lmlist = info
        print(fingers)
        prev_pos, canvas = draw(info, prev_pos, canvas)
        output = sendToAI(gemini_model,canvas,fingers)
    img_combines = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)




    # Display the image in a window
    FRAME_WINDOW.image(img_combines,channels="BGR")
    output_text_area.text(output)
    # Keep the window open and update it for each frame; wait for 1 millisecond between frames
    cv2.waitKey(1)




