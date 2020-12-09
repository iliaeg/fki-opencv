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

# rectList = [(startWidth, startHeight, rectHeight, rectWidth),...]
def maskFrame(frame, rectList):
    maskedFrame = np.copy(frame)

    for rect in rectList:
        (startWidth, startHeight, rectWidth, rectHeight) = rect
        endWidth = startWidth + rectWidth
        endHeight = startHeight + rectHeight

        # (startWidth, endWidth, startHeight, endHeight)
        for k in range(startHeight, endHeight):
            for l in range(startWidth, endWidth):
                maskedFrame[k, l] = 0
    return maskedFrame

def filterKeyPoints(keyPoints, rectList):
    filteredKeyPoints = []
    for kp in keyPoints:
        # print(kp.pt[0], kp.pt[1])
        onTheEdge = False
        for rect in rectList:
            r = kp.size# / 2
            if (rect[0] - r <= kp.pt[0] and kp.pt[0] <= rect[0] + rect[2] + r and
                rect[1] - r <= kp.pt[1] and kp.pt[1] <= rect[1] + rect[3] + r):
                onTheEdge = True
        if not onTheEdge:
            filteredKeyPoints.append(kp)
    return filteredKeyPoints

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

# highThreshold = 150
# lowThreshold = 50

# # for task2
# rho = [10, 5, 1] #10/5/1 # 10 - less red lines; 1 - more lines
# theta = [15, 5, 1] #15/5/1 # 15 - no edges; 5 - only vertical edges; 1 - more edges and more angles
# threshold = 10

paths = {}
addPath(paths, 'init', 'every 10 frame')
addPath(paths, 'task5_1', 'every 10 masked frame')
addPath(paths, 'task5_2', 'every 10 masked frame with SIFT')
addPath(paths, 'task5_3', 'every 10 masked frame with filtered SIFT')

# task5_1
rectList = \
            [
                # initial
                (130,362,52,52),
                (138,24,256,49),
                (459,23,231,51),
                (551,80,159,292),
                (406,375,267,75),
                (226,374,143,52),
                (173, 380, 15, 47),
                (137, 79, 139, 103),
                (284, 106, 10, 21),

                # complex
                (408, 79, 14, 20),
                (426, 81, 13, 18),
                (443, 79, 14, 20),
                (461, 80, 14, 20),
                (497, 79, 31, 20),
                (317, 133, 14, 20),
                (335, 133, 14, 20),
                (353, 133, 14, 20),
                (371, 133, 14, 20),
                (407, 133, 5, 19),
                (411, 133, 10, 6),
                (425, 133, 12, 5),
                (429, 137, 4, 15),

                # additions
                (530, 81, 18, 72),
                (187, 380, 4, 47),
                (496, 274, 15, 7),
                (514, 268, 12, 19),
                (335, 160, 50, 20),

                # horizontal lines
                # (709, 234, 32, 4),
                # (108, 234, 149, 4),
                # (267, 234, 14, 4),
                # (292, 234, 14, 4),
                # (316, 234, 13, 4),
                # (340, 234, 13, 4),
                # (365, 234, 13, 4),
                # (390, 234, 13, 4),
                # (415, 234, 13, 4),
                # (440, 234, 13, 4),
                # (465, 234, 13, 4),
                # (490, 234, 13, 4),
                # (515, 234, 13, 4),
                # (540, 234, 13, 4),
                # all in one
                (0, 234, 848, 4),

                # vertical lines
                # (418, 27, 4, 15),
                # (418, 53, 4, 14),
                # (419, 104, 4, 13),
                # (419, 128, 4, 15),
                # (419, 155, 4, 13),
                # (418, 180, 4, 12),
                # (418, 205, 4, 13),
                # (418, 230, 4, 14),
                # (418, 255, 4, 13),
                # (418, 280, 3, 13),
                # (418, 306, 3, 13),
                # (418, 332, 3, 13),
                # (418, 357, 3, 13),
                # all in one
                (418, 0, 3, 464),
            ]

# clear log file
open(outputDir + "task_5_log.txt", "w").close()
lineFormat = '{:>12}  {:>28}  {:>33} \n'

with open(outputDir + "task_5_log.txt", "a") as logFile:
    logFile.write(
        lineFormat.format(
            "номер кадра",
            "кол-во обнаружено признаков",
            "кол-во отфильтрованных признаков"
        )
    )

i = 0
# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        # display frame
        # cv2.imshow('Frame', frame)

        # task5_1: masking
        maskedFrame = maskFrame(frame, rectList)

        # task5_2: detect sift key points
        gray = cv2.cvtColor(maskedFrame, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keyPoints = sift.detect(gray, None)
        maskedFrameWithKeyPoints = cv2.drawKeypoints(gray, keyPoints, maskedFrame)

        # task5_3: filter sift key points
        filteredKeyPoints = filterKeyPoints(keyPoints, rectList)
        maskedFrameWithFilteredKeyPoints = cv2.drawKeypoints(gray, filteredKeyPoints, maskedFrame)

        # cv2.imshow('maskedFrame', maskedFrame)
        # cv2.waitKey(0)

        # write every 10 frame
        if i % 10 == 0:
            cv2.imwrite(paths['init'] + f'{i}.jpeg', frame)
            cv2.imwrite(paths['task5_1'] + f'{i}.jpeg', maskedFrame)
            cv2.imwrite(paths['task5_2'] + f'{i}.jpeg', maskedFrameWithKeyPoints)
            cv2.imwrite(paths['task5_3'] + f'{i}.jpeg', maskedFrameWithFilteredKeyPoints)

            # append log file
            # (append file every time to be sure that data won't be lost)
            with open(outputDir + "task_5_log.txt", "a") as logFile:
                logFile.write(
                    lineFormat.format(
                        i,
                        len(keyPoints),
                        len(filteredKeyPoints)
                    )
                )

        # break

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else: 
        break

    i += 1

# logFile.close()

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

end_time = time.time()
print(f"Done!\n" +
    f"End time: {datetime.fromtimestamp(end_time)}.\n" +
    f"Total time: {(end_time - start_time) / 60} minutes.")