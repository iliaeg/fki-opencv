import numpy as np
import cv2
import os
from pathlib import Path

import time
from datetime import datetime

start_time = time.time()
print(f"Start time: {datetime.fromtimestamp(start_time)}")

def addPath(pathArray, key, dirName):
    pathArray[key] = os.path.join(outputDir, './' + dirName + '/')
    Path(pathArray[key]).mkdir(parents=True, exist_ok=True)

class Target2D:
    def __init__(self, targetCenter, angleDeg = 0, scale = 150):
        self.targetCenter = targetCenter
        self.angleDeg = angleDeg
        self.scale = scale

    def PrepareModel(self, vertex):
        self.mdlVertex = np.copy(vertex)

    def DrawTarget(self, frame):
        theta = (self.angleDeg/180.) * np.pi
        rotMatrix = np.array(
            [[np.cos(theta), -np.sin(theta)], 
            [np.sin(theta),  np.cos(theta)]]
        )
        points = np.dot(self.mdlVertex * (1, -1), rotMatrix)
        points = points * self.scale + self.targetCenter
        points = np.array(points, np.int32)
        points = points.reshape((-1,1,2))
        cv2.polylines(frame, [points], True, (0,255,255))

# CONSTS
# Relative paths to directories
INPUT_DIR = '../input/videos/'
OUTPUT_DIR = '../output/'
VIDEO_NAME = 'dockingISS.mp4'
# IMAGE_NAME = 'mitsubishi_outlander_on_road.jpg'

dirName = os.path.dirname(__file__)
currentFileName = os.path.splitext(os.path.basename(__file__))[0]
inputPath = os.path.join(dirName, INPUT_DIR + VIDEO_NAME)
outputDir = os.path.join(dirName, OUTPUT_DIR)

# Create output dir if it does not exist
Path(outputDir).mkdir(parents=True, exist_ok=True)

if not os.path.exists(inputPath):
    print(f"File '{VIDEO_NAME}' does not exist; (path: {inputPath})")

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture(inputPath)

# Check if file opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")

paths = {}
addPath(paths, 'task_target', 'target on frame')

mdlVertex = \
    [
        (0,1),
        (0.718, 0.053),
        (0.718, -0.053),
        (0.275, -0.664),
        (-0.275, -0.664),
        (-0.718, -0.053),
        (-0.718, 0.053)
    ]

i = 0
# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        frame = cv2.imread(os.path.join(dirName, "../output/every 10 frame/2300.jpeg"))

        # task_target: draw target on frame
        targetedFrame = np.copy(frame)

        target2 = Target2D((433, 240), 0, 51)
        target2.PrepareModel(mdlVertex)
        target2.DrawTarget(targetedFrame)

        cv2.imshow('targetedFrame', targetedFrame)
        cv2.waitKey(0)

        # write every 10 frame
        if i % 10 == 0:
            cv2.imwrite(paths['task_target'] + f'{i}.jpeg', targetedFrame)

        # break the loop after first frame
        break

        # Press Q on keyboard to exit
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

end_time = time.time()
print(f"Done!\n" +
    f"End time: {datetime.fromtimestamp(end_time)}.\n" +
    f"Total time: {(end_time - start_time) / 60} minutes.")