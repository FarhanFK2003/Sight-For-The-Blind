# Main Code for Camera Capture

from ultralytics import YOLO            # Import the YOLO object detection model from the ultralytics library
from matplotlib import pyplot as plt    # Import pyplot for displaying images with bounding boxes
import cv2                              # OpenCV library for capturing images and video streams
from gtts import gTTS                   # Google Text-to-Speech (gTTS) for converting text to speech
from playsound import playsound         # Playsound for playing audio files
import os                               # Operating system module for file handling

# Load the YOLOv8 model for object detection
model = YOLO('yolov8n.pt')

# Function to capture an image from the camera
def capture_image_from_camera():
    cap = cv2.VideoCapture(0)  # Open the default camera (camera ID 0)

    if not cap.isOpened():  # Check if the camera opened successfully
        print("Error: Could not open video stream from the camera.")
        return None

    print("Press 's' to capture an image, or 'q' to quit.")

    while True:
        ret, frame = cap.read()  # Capture a frame from the camera
        if not ret:  # If the frame couldn't be captured
            print("Error: Failed to capture image.")
            break

        # Display the captured frame in a window named 'Camera'
        cv2.imshow('Camera', frame)
        key = cv2.waitKey(1)  # Wait for a key press with a delay of 1ms

        if key == ord('s'):  # If 's' is pressed, save the current frame as an image
            image_path = "captured_image.jpg"
            cv2.imwrite(image_path, frame)  # Save the image as 'captured_image.jpg'
            print("Image captured and saved as 'captured_image.jpg'.")
            break
        elif key == ord('q'):  # If 'q' is pressed, exit the loop without saving an image
            print("Exiting without capturing image.")
            break

    cap.release()  # Release the camera resource
    cv2.destroyAllWindows()  # Close the camera window

    return image_path  # Return the saved image path


# Capture an image from the camera
image_path = capture_image_from_camera()

# If an image was captured successfully
if image_path is not None:
    # Use the YOLO model to predict objects in the captured image
    results = model.predict(image_path, conf=0.5)  # Set confidence threshold to 0.5

    # Initialize a dictionary to store the count of detected objects
    object_counts = {}

    # Loop through each prediction result
    for r in results:
        # Get the class names of the detected objects
        detected_classes = [r.names[int(c)] for c in r.boxes.cls]

        # Count the number of times each object class is detected
        for obj in detected_classes:
            if obj in object_counts:
                object_counts[obj] += 1  # Increment the count if the object is already detected
            else:
                object_counts[obj] = 1  # Initialize the count if it's the first detection

        # Plot the detected objects with bounding boxes on the image
        r = r.plot(conf=False)  # Plot the bounding boxes without confidence score display
        r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)  # Convert the image from BGR to RGB for displaying
        plt.imshow(r)  # Display the image using matplotlib
        plt.show()

    # If any objects were detected
    if object_counts:
        # Create a string summarizing the detected objects and their counts
        detected_str = ", ".join([f"{count} {obj}" for obj, count in object_counts.items()])
        print(detected_str)  # Print the detected objects and their counts

        # Convert the detected objects summary into speech using gTTS
        tts = gTTS(text=f"Detected objects are: {detected_str}", lang='en')
        tts.save("detected_objects.mp3")  # Save the speech as an mp3 file

        # Play the generated speech
        playsound("detected_objects.mp3")


# Code for Scalability of the Captured video in Real Time

from ultralytics import YOLO            # Import the YOLO object detection model from the ultralytics library
from matplotlib import pyplot as plt    # Import pyplot for displaying images with bounding boxes
import cv2                              # OpenCV for handling video and image capture
from gtts import gTTS                   # Google Text-to-Speech (gTTS) for converting text to speech
from playsound import playsound         # Playsound for playing audio files
import os                               # Operating system module for file handling
import random                           # Random module for selecting a random frame from the video

# Load the pre-trained YOLOv8 model for object detection
model = YOLO('yolov8n.pt')

# Function to process a random frame from a given video file
def process_random_frame_from_video(video_path):
    cap = cv2.VideoCapture(video_path)  # Open the video file

    if not cap.isOpened():  # Check if the video file was opened successfully
        print("Error: Could not open video file.")
        return None

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Select a random frame number from the video
    random_frame = random.randint(0, total_frames - 1)

    print(f"Randomly selected frame number: {random_frame}")

    # Jump to the randomly selected frame in the video
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
    ret, frame = cap.read()  # Capture the frame at the random position

    if not ret:  # Check if the frame was captured successfully
        print("Error: Failed to capture frame.")
        cap.release()
        return None

    # Run the YOLOv8 model to detect objects in the selected frame
    results = model.predict(frame, conf=0.5)  # Set confidence threshold to 0.5

    # Dictionary to store detected objects and their counts
    object_counts = {}

    # Loop through each prediction result
    for r in results:
        # Get the class names of the detected objects
        detected_classes = [r.names[int(c)] for c in r.boxes.cls]

        # Count the number of times each object class is detected
        for obj in detected_classes:
            if obj in object_counts:
                object_counts[obj] += 1  # Increment the count if the object has been detected before
            else:
                object_counts[obj] = 1  # Initialize the count if it's the first detection

        # Plot and display the frame with bounding boxes around detected objects
        r = r.plot(conf=False)  # Plot the bounding boxes without confidence score display
        r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)  # Convert the image from BGR to RGB for displaying
        plt.imshow(r)  # Display the image using matplotlib
        plt.show()

    cap.release()  # Release the video file resource

    return object_counts  # Return the dictionary of detected objects and their counts


# User is prompted to input the path to the video file
video_path = input("Please enter the path to the video file: ")

# Check if the provided video file path exists
if os.path.exists(video_path):
    # Process a random frame from the video and get detected object counts
    object_counts = process_random_frame_from_video(video_path)

    # If objects were detected
    if object_counts:
        # Create a summary string of detected objects and their counts
        detected_str = ", ".join([f"{count} {obj}" for obj, count in object_counts.items()])
        print(f"Detected objects are: {detected_str}")

        # Convert the detected objects summary to speech using gTTS
        tts = gTTS(text=f"Detected objects are: {detected_str}", lang='en')
        tts.save("detected_objects.mp3")  # Save the speech as an mp3 file

        # Play the generated speech
        playsound("detected_objects.mp3")
else:
    # If the provided video file path does not exist, display an error message
    print("Error: The video file path does not exist.")
