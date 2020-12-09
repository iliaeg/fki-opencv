import numpy as np
import cv2
import os
from pathlib import Path

def addPath(pathArray, key, dirName):
    pathArray[key] = os.path.join(outputDir, './' + dirName + '/')
    Path(pathArray[key]).mkdir(parents=True, exist_ok=True)

# CONSTS
# Relative paths to directories
INPUT_DIR = '../input/videos/'
OUTPUT_DIR = '../output/'
VIDEO_NAME = '20191119_1241_Cam_1_03_00.avi'
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

# otsuThreshold = cv2.threshold(grayscaleFrame, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# (img,127,255,cv.THRESH_TOZERO_INV)
# );
highThreshold = 150
lowThreshold = 50

# for task2
rho = [10, 5, 1] #10/5/1 # 10 - less red lines; 1 - more lines
theta = [15, 5, 1] #15/5/1 # 15 - no edges; 5 - only vertical edges; 1 - more edges and more angles
threshold = 10

paths = {}
addPath(paths, 'task1', 'first 50 images')
addPath(paths, 'task2_ex1', 'every 10 contour')
# path['task1'] = os.path.join(outputDir, './every 50 image/')
# Path(path['task1']).mkdir(parents=True, exist_ok=True)

i = 0
# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # task1
        if (i < 50):
            cv2.imwrite(paths['task1'] + str(i) + '.jpeg', frame)

        # task2_ex1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, lowThreshold, highThreshold, 3)

        # write every 10 contour
        if i % 10 == 0:
            cv2.imwrite(paths['task2_ex1'] + f'{i}.jpeg', canny)

        # task2_ex2
        lines = cv2.HoughLines(edges, rho, theta, threshold, min_theta=0, max_theta=90)

        # if (i < 1):
        #     grayscaleFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #     # cv2.imshow('Gray image', grayscaleFrame)
        #     # cv2.imwrite(outputDir + str(i) + '.jpeg', gray_image)
        #     # contours, hierarchy = cv2.findContours(grayscaleFrame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #     # otsuThreshold = cv2.threshold(grayscaleFrame, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #     # highThreshold = otsuThreshold[0]
        #     # lowThreshold = otsuThreshold[0] * 0.5

        #     edges = cv2.Canny(grayscaleFrame, lowThreshold, highThreshold)#threshold,threshold*2)
        #     # print(edges)
        #     drawing = np.zeros(grayscaleFrame.shape,np.uint8)  
        #     contours,hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #     for cnt in contours:
        #         x,y,w,h = cv2.boundingRect(cnt)
        #         cv2.rectangle(grayscaleFrame,(x,y),(x+w,y+h),(0,255,0),2)
        #         rect = cv2.minAreaRect(cnt)
        #         # box = cv2.BoxPoints(rect)
        #         # box = np.int0(box)

        #     cv2.imshow('Frame', grayscaleFrame)
        #     cv2.waitKey(0)

        #     lines = cv2.HoughLines(edges, rho, theta, threshold, min_theta=0, max_theta=90)#1,np.pi/180,200)#, 

        #     for line in lines:
        #         # The below for loop runs till r and theta values
        #         # are in the range of the 2d array
        #         for r,theta in line:#line
        #             # Stores the value of cos(theta) in a
        #             a = np.cos(theta)
        #             # Stores the value of sin(theta) in b
        #             b = np.sin(theta)
        #             # x0 stores the value rcos(theta)
        #             x0 = a*r
        #             # y0 stores the value rsin(theta)
        #             y0 = b*r
        #             # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        #             x1 = int(x0 + 1000*(-b))
        #             # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        #             y1 = int(y0 + 1000*(a))
        #             # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        #             x2 = int(x0 - 1000*(-b))
        #             # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        #             y2 = int(y0 - 1000*(a))
        #             # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
        #             # (0,0,255) denotes the colour of the line to be
        #             #drawn. In this case, it is red.
        #             cv2.line(frame,(x1,y1), (x2,y2), (0,0,255), 2)

        #     # # print(lines)
        #     # for line in lines:
        #     #     # print(line)
        #     #     cv2.line(grayscaleFrame, line, (255, 0, 0), 5)
        #     # # cv2.drawContours(grayscaleFrame, contours, -1, (0,255,0), 3)

        #     # cv2.imshow('Frame', grayscaleFrame)
        #     cv2.imshow('Frame', frame)

        #     # input("Press Enter to continue...")
        #     # print("frame")
        #     cv2.waitKey(0)
        #     break
        #     # cv2.imwrite(outputDir + str(i) + '.jpeg', frame)

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