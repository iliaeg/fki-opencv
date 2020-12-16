import numpy as np
import math
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

class Masker:
    def __init__(self, rectList):
        self.rectList = rectList

    def MaskFrame(self, frame):
        maskedFrame = np.copy(frame)

        for rect in self.rectList:
            (startWidth, startHeight, rectWidth, rectHeight) = rect
            endWidth = startWidth + rectWidth
            endHeight = startHeight + rectHeight

            # (startWidth, endWidth, startHeight, endHeight)
            for k in range(startHeight, endHeight):
                for l in range(startWidth, endWidth):
                    maskedFrame[k, l] = 0
        return maskedFrame

# def vertexToKeyPoints(localKeyPoints):

class Target2D:
    # default parameters for 2300 frame
    def __init__(self, targetCenter = (433, 240), angleDeg = 0, scale = 60):
        self.targetCenter = targetCenter
        self.angleDeg = angleDeg
        self.scale = scale

    # vertices is an array of target vertices counterclockwise starting from the top
    def PrepareModel(self, vertices):
        self.mdlVertex = np.copy(vertices)

    def AddKeyPoints(self, keyPoints, des):
        self.kp = keyPoints
        self.des = des

    def VertexToGlobal(self):
        theta = (self.angleDeg/180.) * np.pi
        rotMatrix = np.array(
            [[np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]]
        )
        points = np.dot(self.mdlVertex * (1, -1), rotMatrix)
        points = points * self.scale + self.targetCenter
        points = np.array(points, np.int32)
        return points

    def VertexToGlobalCropped(self):
        points = self.VertexToGlobal()
        xMin, xMax, yMin, yMax = self.GetMinMaxGlobalVertices()
        points -= (xMin, yMin)
        print("VertexToGlobalCropped:", points)
        return points

    def DrawTarget(self, frame):
        points = self.VertexToGlobal().reshape((-1,1,2))
        cv2.polylines(frame, [points], True, (0,255,255))

    def PointInTarget(self, point):
        x = point[0]
        y = point[1]
        globalVertex = self.VertexToGlobal()
        # if (y < globalVertex[0,1]) or (y > globalVertex[3,1]):
        #     return False
        # elif (x < globalVertex[1,0]) or (x > globalVertex[6,0]):
        #     return False
        if np.cross((x,y) - globalVertex[0], globalVertex[1] - globalVertex[0]) < 0 or \
            np.cross((x,y) - globalVertex[2], globalVertex[3] - globalVertex[2]) < 0 or \
            np.cross(point - globalVertex[4], globalVertex[5] - globalVertex[4]) < 0 or \
            np.cross(point - globalVertex[6], globalVertex[0] - globalVertex[6]) < 0:
            return False
        else:
            return True

    def GetMinMaxGlobalVertices(self):
        globalVertex = self.VertexToGlobal()
        yMin = globalVertex[0,1]
        yMax = globalVertex[3,1]
        xMin = globalVertex[1,0]
        xMax = globalVertex[6,0]
        return xMin, xMax, yMin, yMax

    def Shape(self):
        xMin, xMax, yMin, yMax = self.GetMinMaxGlobalVertices()
        return (yMax - yMin, xMax - xMin)


    def MaskOutOfTarget(self, frame):
        maskedFrame = np.zeros(frame.shape, frame.dtype)

        # get points that belongs to target
        xMin, xMax, yMin, yMax = self.GetMinMaxGlobalVertices()
        for x in range(xMin, xMax):
            for y in range(yMin, yMax):
                if self.PointInTarget((x, y)):
                    maskedFrame[y, x] = frame[y, x]

        # crop frame
        maskedFrame = maskedFrame[yMin:yMax, xMin:xMax]

        # cv2.imshow('MaskOutOfTarget', maskedFrame)
        # cv2.waitKey(0)
        return maskedFrame

class KeyPointsDetector:
    def __init__(self, rectList):
        self.rectList = rectList

    # returns filtered key points
    def Detect(self, maskedFrame):
        gray = cv2.cvtColor(maskedFrame, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keyPoints = sift.detect(gray, None)
        filteredKeyPoints = self.FilterKeyPoints(keyPoints)
        filteredKeyPoints, des = sift.compute(gray, filteredKeyPoints)
        return filteredKeyPoints, des
        # maskedFrameWithKeyPoints = cv2.drawKeypoints(gray, keyPoints, maskedFrame)

    def FilterKeyPoints(self, keyPoints):
        filteredKeyPoints = []
        for kp in keyPoints:
            # print(kp.pt[0], kp.pt[1])
            onTheEdge = False
            for rect in self.rectList:
                r = kp.size / 2
                if (rect[0] - r <= kp.pt[0] and kp.pt[0] <= rect[0] + rect[2] + r and
                    rect[1] - r <= kp.pt[1] and kp.pt[1] <= rect[1] + rect[3] + r):
                    onTheEdge = True
            if not onTheEdge:
                filteredKeyPoints.append(kp)
        return filteredKeyPoints

    def DrawKeyPoints(self, frame, keyPoints):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frameWithKeyPoints = cv2.drawKeypoints(gray, keyPoints, frame)
        return frameWithKeyPoints
        # cv2.imshow('maskedFrame', maskedFrame)
        # cv2.waitKey(0)

class TargetDetector2D:
    def __init__(self, modelTarget2D):
        self.modelTarget2D = modelTarget2D

    def PrepareFlann(self):
        # Get flann matcher
        FLANN_INDEX_KDTREE = 0
        indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        searchParams = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams, searchParams)

    def DetectTarget2D(self, frame):
        maskedFrame = masker.MaskFrame(frame)
        kp, des = kpDetector.Detect(maskedFrame)

        # Matching descriptor vectors with a FLANN based matcher
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        knn_matches = matcher.knnMatch(self.modelTarget2D.des, des, 2)
        #-- Filter matches using the Lowe's ratio test
        ratio_thresh = 0.7
        good_matches = []
        for m,n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        #-- Draw matches
        img_matches = np.empty((max(self.modelTarget2D.Shape()[0], frame.shape[0]), self.modelTarget2D.Shape()[1]+frame.shape[1], 3), dtype=np.uint8)
        cv2.drawMatches(
            modelTargetFrame,
            self.modelTarget2D.kp,
            maskedFrame,
            kp,
            good_matches,
            img_matches,
            matchColor=(0, 255, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        img_object = modelTargetFrame
        keypoints_obj = self.modelTarget2D.kp
        img_scene = maskedFrame
        keypoints_scene = kp

        #-- Localize the object
        obj = np.empty((len(good_matches),2), dtype=np.float32)
        scene = np.empty((len(good_matches),2), dtype=np.float32)
        for i in range(len(good_matches)):
            #-- Get the keypoints from the good matches
            obj[i,0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
            obj[i,1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
            scene[i,0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
            scene[i,1] = keypoints_scene[good_matches[i].trainIdx].pt[1]
        H, _ =  cv2.findHomography(obj, scene, cv2.RANSAC)
        # print("[self.modelTarget2D.VertexToGlobal().reshape((-1,1,2))]:", cv2.UMat(self.modelTarget2D.VertexToGlobal().reshape((-1,1,2))))
        # dst = cv2.perspectiveTransform(cv2.UMat([self.modelTarget2D.VertexToGlobal().reshape((-1,1,2))], (7,2)), H)
        self.modelTarget2D.VertexToGlobalCropped()
        print("self.modelTarget2D.mdlVertex:", self.modelTarget2D.VertexToGlobal())
        dst = cv2.perspectiveTransform(np.float32(self.modelTarget2D.VertexToGlobalCropped()).reshape(-1,1,2), H)
        globalVertex = np.int32(dst).reshape(7,2)
        radian = math.atan2(abs(globalVertex[0,0] - targetCenter[0]), abs(globalVertex[0,1]-targetCenter[1]))
        radian2 = math.atan2(abs(241-136), abs(448-382))
        radian1 = math.atan2(abs(241-144), abs(448-547))
        degree2 = math.degrees(radian2)
        degree1 = math.degrees(radian1)
        degree = math.degrees(radian)
        # angle = 
        # scale = globalVertex[0,1] - 
        print("H:",H)
        print("_:",_)

        # #-- Get the corners from the image_1 ( the object to be "detected" )
        # obj_corners = np.empty((4,1,2), dtype=np.float32)
        # obj_corners[0,0,0] = 0
        # obj_corners[0,0,1] = 0
        # obj_corners[1,0,0] = img_object.shape[1]
        # obj_corners[1,0,1] = 0
        # obj_corners[2,0,0] = img_object.shape[1]
        # obj_corners[2,0,1] = img_object.shape[0]
        # obj_corners[3,0,0] = 0
        # obj_corners[3,0,1] = img_object.shape[0]
        # scene_corners = cv2.perspectiveTransform(obj_corners, H)
        # #-- Draw lines between the corners (the mapped object in the scene - image_2 )
        # cv2.line(img_matches, (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])),\
        #     (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])), (0,255,0), 4)
        # cv2.line(img_matches, (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])),\
        #     (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])), (0,255,0), 4)
        # cv2.line(img_matches, (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])),\
        #     (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])), (0,255,0), 4)
        # cv2.line(img_matches, (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])),\
        #     (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])), (0,255,0), 4)
        # #-- Show detected matches
        # cv2.imshow('Good Matches & Object detection', img_matches)
        # cv2.waitKey(0)

        # for point in self.modelTarget2D.kp:
        #     h, status = cv2.findHomography(self.modelTarget2D.kp.pt, kp.pt)
        #     print("h:", h)
        #     print("status:",status)

        # if len(good_matches)>MIN_MATCH_COUNT:
        #     for m in good_matches:
        #         src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        #         dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        #         M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        #         matchesMask = mask.ravel().tolist()
        #         h,w,d = img1.shape
        #         pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        #         dst = cv.perspectiveTransform(pts,M)
        #         img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
        # else:
        #     print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        #     matchesMask = None

        return frame, Target2D(), len(good_matches), img_matches
        # return targetedFrame, target2D, kpFound, flannFrame

        # matches = self.flann.knnMatch(self.modelTarget2D.des, des, k=2)
        # print("matches:", matches)
        # # Prepare an empty mask Draw a good match
        # matchesMask = [[0, 0] for i in range(len(matches))]
        
        # for i, (m, n) in enumerate(matches):
        #     if m.distance < 0.7*n.distance:
        #         matchesMask[i] = [1, 0]
        
        # drawPrams = dict(matchColor=(0, 255, 0),
        #                 singlePointColor=(255, 0, 0),
        #                 matchesMask=matchesMask,
        #                 flags=0)
        # # Match result image
        # flannFrame = cv2.drawMatchesKnn(modelTargetFrame, self.modelTarget2D.kp, maskedFrame, kp, matches, None, **drawPrams)

        # cv2.imshow("matches", img3)
        # cv2.waitKey()

        # return frame, Target2D(), len(matches), flannFrame
        # return targetedFrame, target2D, kpFound, flannFrame




        # targetCenter, angleDeg, scale, keyPoints = \
        #     FeatureСomparison(
        #         self.frame,
        #         keyPointsToGlobal(self.modelTarget2D.keyPoints)
        #     )

        # target2D = new Target2D(keyPoints, targetCenter, angleDeg, scale)
        # target2D.PrepareModel() # Рассчитываем массив mdlVertex

        # return target2D

    # def FeatureСomparison(frame):
    #     # Получет на вход новый кадр и массив SIFT признаков
    #     # эталонного изображения в СК, привязанной к кадру
    #     # (заранее переводим keyPoints в globalKeyPoints)

    #     # Находит SIFT признаки на кадре
    #     # и сопоставляет их с эталонными SIFT признаками
    #     # <Использовать результат 8 задачи>

    #     # !!! в идеале вычислим 8 признаков, которые нас интересуют:
    #     # !!! 7 угловых и центральный

    #     # Для дальнейшей обработки используются пары сопоставленных SIFT признаков -
    #     # подмножество точек modelTarget2D keyPoints и соответствующие признаки,
    #     # обнаруженные на текущем кадре frame

    #     # После сопоставления мы можем вычислить масштаб scale
    #     # и центр мишени targetCenter,
    #     # используя координаты центра и вершины мишени,
    #     # а также угол поворота angleDeg, используя ещё одну точку мишени

    #     # Формируем массив keyPoints

    #     # Возвращает найденые ключенвые признаки на изображении,
    #     # а также targetCenter, angleDeg, scale

    #     return targetCenter, angleDeg, scale, keyPoints


    # def keyPointsToGlobal(target2D):
    #     # переводим keyPoints из локальной СК в 
    #     # СК, привязанную к кадру
    #     return globalKeyPoints

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
addPath(paths, 'init', 'every 10 frame')
addPath(paths, 'task9', 'every 10 targeted frame')
addPath(paths, 'task9_flann', 'every 10 flann frame')

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
                (418, 0, 4, 464),
            ]

# modelVertex = \
#     [
#         (0,1),
#         (0.718, 0.053),
#         (0.718, -0.053),
#         (0.275, -0.664),
#         (-0.275, -0.664),
#         (-0.718, -0.053),
#         (-0.718, 0.053)
#     ]

# Bypassing target vertices counterclockwise starting from the top
modelVertex = \
    [
        (0,1),
        (-0.718, 0.053),
        (-0.718, -0.053),
        (-0.275, -0.664),
        (0.275, -0.664),
        (0.718, -0.053),
        (0.718, 0.053),
    ]

# Prepare classes & modelTarget2D
modelFrame = cv2.imread(os.path.join(dirName, "../input/images/2300.jpeg"))

masker = Masker(rectList)
kpDetector = KeyPointsDetector(rectList)

# modelKeyPoints = 
modelTarget2D = Target2D()
modelTarget2D.PrepareModel(modelVertex)

modelMaskedFrame = masker.MaskFrame(modelFrame)
modelTargetFrame = modelTarget2D.MaskOutOfTarget(modelMaskedFrame)
modelKeyPoints, modelDes = kpDetector.Detect(modelTargetFrame)
modelTarget2D.AddKeyPoints(modelKeyPoints, modelDes)

targetDetector2D = TargetDetector2D(modelTarget2D)
targetDetector2D.PrepareFlann()
# cv2.imshow('modelTarget2D.AddKeyPoints', kpDetector.DrawKeyPoints(modelTargetFrame, modelKeyPoints))
# cv2.waitKey(0)

# modelTarget2D.DrawTarget(modelTargetFrame)
# cv2.imshow('modelMaskedFrame', modelTargetFrame)
# cv2.waitKey(0)

# targetDetector2D = targetDetector2D(modelTarget2D)
# targetDetector2D.DetectTarget2D(modelMaskedFrame)

# clear log file
logFileName = "task_9_log.txt"
open(outputDir + logFileName, "w").close()
lineFormat = '{:>12}  {:>27} \n'# {:>33} \n'

with open(outputDir + logFileName, "a") as logFile:
    logFile.write(
        lineFormat.format(
            "номер кадра",
            "кол-во совпавших признаков"#,
            #"кол-во отфильтрованных признаков"
        )
    )

i = 0
# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        if i < 800:
            i += 1
            continue

        # # task9_1: masking
        # maskedFrame = maskFrame(frame, rectList)

        # # task5_2: detect sift key points
        # gray = cv2.cvtColor(maskedFrame, cv2.COLOR_BGR2GRAY)
        # sift = cv2.SIFT_create()
        # keyPoints = sift.detect(gray, None)
        # maskedFrameWithKeyPoints = cv2.drawKeypoints(gray, keyPoints, maskedFrame)

        # # task5_3: filter sift key points
        # filteredKeyPoints = filterKeyPoints(keyPoints, rectList)
        # maskedFrameWithFilteredKeyPoints = cv2.drawKeypoints(gray, filteredKeyPoints, maskedFrame)

        # cv2.imshow('maskedFrame', maskedFrame)
        # cv2.waitKey(0)

        # task9
        targetedFrame, target2D, kpFound, flannFrame = targetDetector2D.DetectTarget2D(frame)
        cv2.imshow('flannFrame', flannFrame)
        cv2.waitKey(0)

        # append log file
        # (append file every time to be sure that data won't be lost)
        with open(outputDir + logFileName, "a") as logFile:
            logFile.write(
                lineFormat.format(
                    i,
                    kpFound
                )
            )

        # write every 10 frame
        if i % 10 == 0:
            cv2.imwrite(paths['init'] + f'{i}.jpeg', frame)
            cv2.imwrite(paths['task9'] + f'{i}.jpeg', targetedFrame)
            cv2.imwrite(paths['task9_flann'] + f'{i}.jpeg', flannFrame)
            # cv2.imwrite(paths['task5_2'] + f'{i}.jpeg', maskedFrameWithKeyPoints)
            # cv2.imwrite(paths['task5_3'] + f'{i}.jpeg', maskedFrameWithFilteredKeyPoints)

        break

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