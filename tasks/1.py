import numpy
import cv2
import os 

# CONSTS
# Relative paths to directories
INPUT_DIR = '../video/'
OUTPUT_DIR = '../output/'
VIDEO_NAME = '20191119_1241_Cam_1_03_00.avi'

dirname = os.path.dirname(__file__)
inputPath = os.path.join(dirname, INPUT_DIR + VIDEO_NAME)
outputDir = os.path.join(dirname, OUTPUT_DIR)

print(f"{VIDEO_NAME} file exists: {os.path.exists(inputPath)}; (path: {inputPath})")

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(inputPath)

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

i = 0
# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        if (i < 50):
            cv2.imwrite(outputDir + str(i) + '.jpg', frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else: 
        break
    i += 1

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()