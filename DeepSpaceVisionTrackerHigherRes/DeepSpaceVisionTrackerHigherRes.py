import libjevois as jevois
import cv2
import numpy as np
import json
import math
import time
from collections import OrderedDict
import numpy as np

## Detect Vision Targets for Deep Space
#
# Add some description of your module here.
#
# @author Harrison McIntyre
# 
# @videomapping YUYV 320 240 60 YUYV 320 240 60 MARSRovers DeepSpaceVisionTracking
# @email hamac2003@gmail.com
# @address somewhereOnMars
# @copyright Copyright (C) 2018 by Harrison McIntyre
# @mainurl www.team4004.com
# @supporturl www.team4004.com
# @otherurl www.team4004.com
# @license 
# @distribution Unrestricted
# @restrictions None
# @ingroup modules

# The numerical constant pi
pi = 3.14159265358979323846264338327950288416716939937510


# Threshold values for HSV tracking, along with other parameters
# such as dialation and erosion values.
uh = 180
lh = 10
us = 255
ls = 10
uv = 255
lv = 100
er = 0
dl = 0
ap = 6
minar = 1
maxar = 300000
sl = 1.0

# Values used to determine whether the aspect ratio of
# each rectange is within an acceptable tolerence
lowerRatioBound = 2
upperRatioBound = 5 




# A class to hold our global variables
class Vars():
    # The number of frames we wait before declaring a target as "lost"
    lostFrames = 20

    # Keeps track of how long it has been since we have seen a target set
    noTargetCounter = 0
    
    # X value to be sent to roboRio
    xToFollow = None
    yToFollow = None
    frameCount = 0

    # Variable to hold our target data to be serialized
    pixels = 0
    pixels2 = 0



# A class to represent our final target sets which consist of
# two rectangles angled towards each other
class targetSet():
    x = 0
    y = 0
    shapes = []
    boxes = []
    def __init__(self, x, y, shapes, boxes):
        self.x = x
        self.y = y
        self.shapes = shapes
        self.boxes = boxes

# A class to represent our subtargets
# i.e. Individual rectangles that have yet to be
# assigned to a target set
class subTarget():
    x = 0
    y = 0
    width = 0
    height = 0
    angle = 0
    linex1 = 0
    liney1 = 0
    linex2 = 0
    liney2 = 0
    lineLength = 0
    shape = 0

    # Initialization of our subtargets
    def __init__(self, x, y, width, height, angle, shape, box, centerx, centery):
        self.x = centerx
        self.y = centery
        self.linex1 = x
        self.liney1 = y
        self.width = width
        self.height = height
        self.angle = angle
        self.shape = shape
        self.box = box
    
    # Calculate the two points of each of each of the line segments we are
    # going to draw coming out of each of our subtargets
    def calculateLine(self):
        
        if self.height > self.width:
            self.lineLength = 7*self.height
        else:
            self.lineLength = 7*self.width

        self.linex2 = int(round(self.linex1 + self.lineLength * math.cos(self.angle * pi / 180.0)))
        self.liney2 = int(round(self.liney1 + self.lineLength * math.sin(self.angle * pi / 180.0)))


# A class that holds the functions we use to draw out line segments,
# and to calculate whether any two given rays intersect
class Functions():

    # Clean up the angles returned by OpenCV
    def convertAngle(self, calculatedRectWidth, calculatedRectHeight, calculatedRectAngle):
        if calculatedRectWidth < calculatedRectHeight:
            calculatedRectAngle+=180
        else:
            calculatedRectAngle+=90

        if calculatedRectAngle > 90:
            calculatedRectAngle+=180
        return calculatedRectAngle
    
    # Draw a line segment between points "p1", and "p2"
    def line(self, p1, p2):
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0]*p2[1] - p2[0]*p1[1])
        return A, B, -C, p1, p2
    
    # Determine whether line "L1" intersects line "L2".
    # If it does, return the x,y coordinates of the
    # intersection.
    def intersection(self, L1, L2, lineLength):
        D  = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]
        if D != 0:
            x = Dx / D
            y = Dy / D
            deltaX = x - L1[3][0]
            deltaY = y - L1[3][1]
            dist = math.sqrt(math.pow(deltaX, 2) + math.pow(deltaY, 2))
            if dist > lineLength or deltaY > 0:
                return False
            else:
                return int(round(x)),int(round(y))
        else:
            return False



# Variable to hold an instance of our "Functions" class
functions = Functions()


# The class that runs our main tracking code
class DeepSpaceVisionTrackerHigherRes:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # Instantiate a JeVois Timer to measure our processing framerate:
        self.timer = jevois.Timer("processing timer", 100, jevois.LOG_INFO)
    
    # This function runs in "headless mode", and doesn't provide any frames via USB output
    def processNoUSB(self, inframe):
        
        # Keep track of the number of frames
        Vars.frameCount+=1

        # Get the next camera image (may block until it is captured) and here convert it to OpenCV BGR by default. If
        # you need a grayscale image instead, just use getCvGRAY() instead of getCvBGR(). Also supported are getCvRGB()
        # and getCvRGBA():
        inimg = inframe.getCvBGR()
        outimg = inimg

        # Start measuring image processing time (NOTE: does not account for input conversion time):
        self.timer.start()

        #Convert the image from BGR(RGB) to HSV
        hsvImage = cv2.cvtColor(inimg, cv2.COLOR_BGR2HSV)
        
        # Threshold HSV Image to find specific color
        binImage = cv2.inRange(hsvImage, (lh, ls, lv), (uh, us, uv))
        
        # Erode image to remove noise if necessary.
        binImage = cv2.erode(binImage, None, iterations = er)

        # Dilate image to fill in gaps
        binImage = cv2.dilate(binImage, None, iterations = dl)
        
        # Finds contours (like finding edges/sides), 'contours' is what we are after
        contours, im2 = cv2.findContours(binImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
        
        # Variables
        rects = []
        badPolys = []
        finalShapes = []

        angles = []
        points = []
        finalCrossPoints = []

        a = 0
        b = 0
        c = 0

        subTargets = []
        midTargets = []
        finalTargets = []
        
        result = True

        preValue = 1000

        placeHoldingTarget = None
        
        # Parse through contours to find targets
        for c in contours:
            if (contours != None) and (len(contours) > 0):
                cnt_area = cv2.contourArea(c)
                hull = cv2.convexHull(c , 1)
                hull_area = cv2.contourArea(hull)  # Used in Solidity calculation
                p = cv2.approxPolyDP(hull, ap, 1)
                if (cv2.isContourConvex(p) != False) and (len(p) >= 4) and (cv2.contourArea(p) >= minar) and (cv2.contourArea(p) <= maxar):
                    filled = cnt_area/hull_area
                    if filled <= sl:
                        rects.append(p)
                else:
                    badPolys.append(p)
        
        # If the camera can see at least 2 rectangles, determine whether
        # there is a target set visible
        if len(rects) > 1:
            ratio = 0
            # For every rectangle in the array "rects", loop once
            for s in rects:
                # Get the area of the current square
                br = cv2.boundingRect(s)
            
                # Find the "x" and "y" coordinates of the center of our shape
                x = br[0] + (br[2]/2)
                y = br[1] + (br[3]/2)
                
                # Find the contour area of our shape
                cnt_area = cv2.contourArea(s)

                # Create a rotated bounding box for our shape
                hull = cv2.convexHull(s , 1)
                hull_area = cv2.contourArea(hull)
                p = cv2.approxPolyDP(hull, ap, 1)
                rect = cv2.minAreaRect(s)
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                # Take the angle of our rectangle, and convert it to a usable form
                angle = functions.convertAngle(rect[1][0],rect[1][1],rect[2])

                # Find the ratio of the longer side of our rectangle, to the shorter side
                if rect[1][0] > rect[1][1]:
                    ratio = rect[1][0]/rect[1][1]
                else:
                    ratio = rect[1][1]/rect[1][0]

                # If the ratio is within an accpetable range, create an instance of our subtarget class
                if ratio >= 2 and ratio <= 5:    
                    target = subTarget(round(rect[0][0]),round(rect[0][1]),rect[1][0],rect[1][1], angle-90, s, box,x,y)
                    target.calculateLine()
                    subTargets.append(target)

            # For every subtarget in our "subtargets" array, draw a line segment over it
            for target in subTargets:
                outimg = cv2.line(outimg,(target.linex1,target.liney1),(target.linex2,target.liney2),(255,255,0),2)

            # Loop through every possible combination of the targets in our "subtargets" array
            for i in range(len(subTargets)):
                for j in range(len(subTargets)):
                    if i != j:
                        # Determine whether or not the line segments for each of our subtargets intersect
                        L1 = functions.line([subTargets[i].linex1, subTargets[i].liney1],[subTargets[i].linex2, subTargets[i].liney2])
                        L2 = functions.line([subTargets[j].linex1, subTargets[j].liney1],[subTargets[j].linex2, subTargets[j].liney2])
            
                        if subTargets[i].lineLength > subTargets[j].lineLength:    
                            R = functions.intersection(L1, L2, subTargets[i].lineLength)
                        else:
                            R = functions.intersection(L1, L2, subTargets[j].lineLength)
                        if R:
                            for t in finalCrossPoints:
                                if R == t:
                                    result = False
                                    break
                                else:
                                    result = True
                            # If the line segments intersect, append that set of targets to our "midTargets" array
                            if result:
                                cv2.circle(outimg,R, 5, (0,0,255), -1)
                                midTargets.append([subTargets[i],subTargets[j]])
                                finalCrossPoints.append(R)
                            else:
                                continue

            # For each target set in our "midTargets" array, loop once
            for target in midTargets:
                #cv2.drawContours(outimg,[finalTargets[0][0].box],0,(0,0,255),2)
                #cv2.drawContours(outimg,[finalTargets[0][1].box],0,(0,0,255),2)
                
                # Find the x-y coordinates of our target set
                leftX = target[0].x
                rightX = target[1].x
                leftY = target[0].y
                rightY = target[1].y
                if leftX > rightX:
                    tempX = leftX
                    leftX = rightX
                    rightX = tempX
                    
                    tempY = leftY
                    leftY = rightY
                    rightY = tempY
                    

                x = int(((rightX-leftX)/2)+leftX)
                y = int(((rightY-leftY)/2)+leftY)
                

                # Create and instance of our "targetSet" class
                myTarget = targetSet(x, y, [target[0].shape, target[1].shape], [target[0].box, target[1].box])
                #cv2.rectangle(outimg,(x,y),(x+w,y+h),(0,255,0),2)
                ##cv2.rectangle(outimg, (br[0],br[1]),((br[0]+br[2]),(br[1]+br[3])),(0,0,255), 2,cv2.LINE_AA)
                #Vars.pixels = x

                # Append our targetset to our "finalTargets" array
                finalTargets.append(myTarget)
                #cv2.circle(outimg,(x,y), 10, (0,255,0), -1)
            
            # For every targetset in our "finalTargets" array, loop once
            for target in finalTargets:
                # Draw the bounding rectangles on our final targetsets
                cv2.drawContours(outimg,[target.boxes[0]],0,(0,0,255),2)
                cv2.drawContours(outimg,[target.boxes[1]],0,(0,0,255),2)

                # If we can see more than one targetset, determine which target is closer to the center of the
                # camera, and store that target in the "placeHoldingTarget" variable
                if len(finalTargets) > 1:
                    if abs(320 - target.x) < preValue:   
                        preValue = abs(320 - target.x)
                        placeHoldingTarget = target
                else:
                    preValue = abs(320-target.x)
                    placeHoldingTarget = target

        # Determine whether we have lost the target long enough to consider the target "lost"
        if len(finalTargets) > 0:
            Vars.xToFollow = placeHoldingTarget.x
        elif Vars.noTargetCounter <= Vars.lostFrames:
            Vars.noTargetCounter+=1
        elif Vars.noTargetCounter > Vars.lostFrames:
            Vars.xToFollow = None
            Vars.noTargetCounter = 0

        # Draw a circle at the x-y coordinates of the target we want to follow
        if Vars.xToFollow is not None:
            cv2.circle(outimg,(Vars.xToFollow,240), 10, (0,255,0), -1)
            Vars.pixels = Vars.xToFollow
        else:
            Vars.pixels = -1

        # Serialize the "x" coordinate of the target we want our robot to follow
        json_pixels = json.dumps(Vars.pixels)
        #Send that data to the roboRio
        jevois.sendSerial(json_pixels)
            
       
    # Process function that provides frames via USB output
    def process(self, inframe, outframe):
               # Keep track of the number of frames
        Vars.frameCount+=1

        # Get the next camera image (may block until it is captured) and here convert it to OpenCV BGR by default. If
        # you need a grayscale image instead, just use getCvGRAY() instead of getCvBGR(). Also supported are getCvRGB()
        # and getCvRGBA():
        inimg = inframe.getCvBGR()
        outimg = inimg

        # Start measuring image processing time (NOTE: does not account for input conversion time):
        self.timer.start()

        #Convert the image from BGR(RGB) to HSV
        hsvImage = cv2.cvtColor(inimg, cv2.COLOR_BGR2HSV)
        
        # Threshold HSV Image to find specific color
        binImage = cv2.inRange(hsvImage, (lh, ls, lv), (uh, us, uv))
        
        # Erode image to remove noise if necessary.
        binImage = cv2.erode(binImage, None, iterations = er)

        # Dilate image to fill in gaps
        binImage = cv2.dilate(binImage, None, iterations = dl)
        
        # Finds contours (like finding edges/sides), 'contours' is what we are after
        contours, im2 = cv2.findContours(binImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
        
        # Variables
        rects = []
        badPolys = []
        finalShapes = []

        angles = []
        points = []
        finalCrossPoints = []

        a = 0
        b = 0
        c = 0

        subTargets = []
        midTargets = []
        finalTargets = []
        
        result = True

        preValue = 1000

        placeHoldingTarget = None
        
        # Parse through contours to find targets
        for c in contours:
            if (contours != None) and (len(contours) > 0):
                cnt_area = cv2.contourArea(c)
                hull = cv2.convexHull(c , 1)
                hull_area = cv2.contourArea(hull)  # Used in Solidity calculation
                p = cv2.approxPolyDP(hull, ap, 1)
                if (cv2.isContourConvex(p) != False) and (len(p) >= 4) and (cv2.contourArea(p) >= minar) and (cv2.contourArea(p) <= maxar):
                    filled = cnt_area/hull_area
                    if filled <= sl:
                        rects.append(p)
                else:
                    badPolys.append(p)
        
        # If the camera can see at least 2 rectangles, determine whether
        # there is a target set visible
        if len(rects) > 1:
            ratio = 0
            # For every rectangle in the array "rects", loop once
            for s in rects:
                # Get the area of the current square
                br = cv2.boundingRect(s)
            
                # Find the "x" and "y" coordinates of the center of our shape
                x = br[0] + (br[2]/2)
                y = br[1] + (br[3]/2)
                
                # Find the contour area of our shape
                cnt_area = cv2.contourArea(s)

                # Create a rotated bounding box for our shape
                hull = cv2.convexHull(s , 1)
                hull_area = cv2.contourArea(hull)
                p = cv2.approxPolyDP(hull, ap, 1)
                rect = cv2.minAreaRect(s)
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                # Take the angle of our rectangle, and convert it to a usable form
                angle = functions.convertAngle(rect[1][0],rect[1][1],rect[2])

                # Find the ratio of the longer side of our rectangle, to the shorter side
                if rect[1][0] > rect[1][1]:
                    ratio = rect[1][0]/rect[1][1]
                else:
                    ratio = rect[1][1]/rect[1][0]

                # If the ratio is within an accpetable range, create an instance of our subtarget class
                if ratio >= 2 and ratio <= 5:    
                    target = subTarget(round(rect[0][0]),round(rect[0][1]),rect[1][0],rect[1][1], angle-90, s, box,x,y)
                    target.calculateLine()
                    subTargets.append(target)

            # For every subtarget in our "subtargets" array, draw a line segment over it
            for target in subTargets:
                outimg = cv2.line(outimg,(target.linex1,target.liney1),(target.linex2,target.liney2),(255,255,0),2)

            # Loop through every possible combination of the targets in our "subtargets" array
            for i in range(len(subTargets)):
                for j in range(len(subTargets)):
                    if i != j:
                        # Determine whether or not the line segments for each of our subtargets intersect
                        L1 = functions.line([subTargets[i].linex1, subTargets[i].liney1],[subTargets[i].linex2, subTargets[i].liney2])
                        L2 = functions.line([subTargets[j].linex1, subTargets[j].liney1],[subTargets[j].linex2, subTargets[j].liney2])
            
                        if subTargets[i].lineLength > subTargets[j].lineLength:    
                            R = functions.intersection(L1, L2, subTargets[i].lineLength)
                        else:
                            R = functions.intersection(L1, L2, subTargets[j].lineLength)
                        if R:
                            for t in finalCrossPoints:
                                if R == t:
                                    result = False
                                    break
                                else:
                                    result = True
                            # If the line segments intersect, append that set of targets to our "midTargets" array
                            if result:
                                cv2.circle(outimg,R, 5, (0,0,255), -1)
                                midTargets.append([subTargets[i],subTargets[j]])
                                finalCrossPoints.append(R)
                            else:
                                continue

            # For each target set in our "midTargets" array, loop once
            for target in midTargets:
                #cv2.drawContours(outimg,[finalTargets[0][0].box],0,(0,0,255),2)
                #cv2.drawContours(outimg,[finalTargets[0][1].box],0,(0,0,255),2)
                
                # Find the x-y coordinates of our target set
                leftX = target[0].x
                rightX = target[1].x
                leftY = target[0].y
                rightY = target[1].y
                if leftX > rightX:
                    tempX = leftX
                    leftX = rightX
                    rightX = tempX
                    
                    tempY = leftY
                    leftY = rightY
                    rightY = tempY
                    

                x = int(((rightX-leftX)/2)+leftX)
                y = int(((rightY-leftY)/2)+leftY)
                

                # Create and instance of our "targetSet" class
                myTarget = targetSet(x, y, [target[0].shape, target[1].shape], [target[0].box, target[1].box])
                #cv2.rectangle(outimg,(x,y),(x+w,y+h),(0,255,0),2)
                ##cv2.rectangle(outimg, (br[0],br[1]),((br[0]+br[2]),(br[1]+br[3])),(0,0,255), 2,cv2.LINE_AA)
                #Vars.pixels = x

                # Append our targetset to our "finalTargets" array
                finalTargets.append(myTarget)
                #cv2.circle(outimg,(x,y), 10, (0,255,0), -1)
            
            # For every targetset in our "finalTargets" array, loop once
            for target in finalTargets:
                # Draw the bounding rectangles on our final targetsets
                cv2.drawContours(outimg,[target.boxes[0]],0,(0,0,255),2)
                cv2.drawContours(outimg,[target.boxes[1]],0,(0,0,255),2)

                # If we can see more than one targetset, determine which target is closer to the center of the
                # camera, and store that target in the "placeHoldingTarget" variable
                if len(finalTargets) > 1:
                    if abs(160 - target.x) < preValue:   
                        preValue = abs(160 - target.x)
                        placeHoldingTarget = target
                else:
                    preValue = abs(160-target.x)
                    placeHoldingTarget = target

        # Determine whether we have lost the target long enough to consider the target "lost"
        if len(finalTargets) > 0:
            Vars.xToFollow = placeHoldingTarget.x
            Vars.yToFollow = placeHoldingTarget.y
        elif Vars.noTargetCounter <= Vars.lostFrames:
            Vars.noTargetCounter+=1
        elif Vars.noTargetCounter > Vars.lostFrames:
            Vars.xToFollow = None
            Vars.yToFollow = None
            Vars.noTargetCounter = 0

        # Draw a circle at the x-y coordinates of the target we want to follow
        if Vars.xToFollow is not None:
            cv2.circle(outimg,(Vars.xToFollow,Vars.yToFollow), 10, (0,255,0), -1)
            Vars.pixels = -Vars.xToFollow+320
            Vars.pixels2 = -Vars.yToFollow+240 
        else:
            Vars.pixels = -1
            Vars.pixels2 = -1
        
        # Serialize the "x" coordinate of the target we want our robot to follow
        json_pixels = json.dumps(Vars.pixels)
        other = json.dumps(Vars.pixels2)
        
        finalSend = "T2 " + json_pixels + " " + other
        

        
        
        #Send that data to the roboRio
        jevois.sendSerial(finalSend)
        #outimg = binImage
        # Send the output frame to our computer
        outframe.sendCv(outimg)
