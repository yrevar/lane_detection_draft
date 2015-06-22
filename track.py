#! /usr/bin/env python
import os
import sys
import csv
import cv2
import glob
import numpy as np
import math
import linefn
from matplotlib import pyplot as plt

# Todo conf structures
LANE_ROI_START_Y = 600

MIN_LANE_LENGTH = 150
MIN_LANE_ANGLE = 10

GAUSSIAN_KERNEL_SIZE = (5,5)
GAUSSIAN_KERNEL_SIGX = 0 
GAUSSIAN_KERNEL_SIGY = 0

CLAHE_CLIP_LIMIT=2.0
CLAHE_TILE_GRID_SIZE=(8,8)

CANNY_HYSTERESIS_MAXVAL = 120 
CANNY_HYSTERESIS_MINVAL = 60
CANNY_L2_GRADIENT=True
CANNY_APERTURE_SIZE=3

HOUGH_MIN_VOTES = 100
HOUGH_MIN_LINE_LEN = 120 
HOUGH_MAX_LINE_GAP = 18

LANE_CALIBRATION = { 'lx': 300, 'ly': 438, 'vx': 722, 'vy': 0, 'rx':  1222, 'ry': 438}

DRAW_ALL_HOUGH_LINES = False
DRAW_CALIBRATION = False
DRAW_FILTERED_LANES = False

DISPLAY_FULL_IMAGE=True


def getLaneCalibrationPoints(lane_calibration):
    return (lane_calibration['lx'], lane_calibration['ly'], 
                lane_calibration['vx'], lane_calibration['vy'],
                lane_calibration['rx'], lane_calibration['ry'])

def getLinesInCenterLaneProximity(lines, lane_calibration):
    c1x, c1y, c2x, c2y, c3x, c3y = getLaneCalibrationPoints(lane_calibration)
    left_lane_lines = []    
    right_lane_lines = []
    for x1,y1,x2,y2 in lines:
        m = linefn.getLineSlope(x1,y1,x2,y2)
        angle = math.atan(m)*180/float(math.pi)
        if abs(angle) >= MIN_LANE_ANGLE:
            if (x1+x2)/2.0 <= c2x and (y1+ y2)/2 >= c2y and angle <= -MIN_LANE_ANGLE: 
                left_lane_lines.append((x1,y1,x2,y2))
            elif (x1+x2)/2.0 > c2x and (y1+ y2)/2 >= c2y and angle >= MIN_LANE_ANGLE: 
                right_lane_lines.append((x1,y1,x2,y2))
                
    return (left_lane_lines, right_lane_lines)

def sortLineFcn(line): # TODO write sorting key function with fitting  criteria
    return line[4]
    
def lineInnerProduct(l1x1,l1y1,l1x2,l1y2,l2x1,l2y1,l2x2,l2y2):
    m1x, m1y = (l1x1 + l1x2)/2.0, (l1y1 + l1y2)/2.0
    m2x, m2y = (l2x1 + l2x2)/2.0, (l2y1 + l2y2)/2.0
    angle1 = math.atan(linefn.getLineSlope(l1x1,l1y1,l1x2,l1y2))*180/float(math.pi) 
    angle2 = math.atan(linefn.getLineSlope(l2x1,l2y1,l2x2,l2y2))*180/float(math.pi) 
    return math.sqrt((m1x-m2x)**2 + (m1y-m2y)**2) * math.cos( abs(max(angle1, angle2)) - abs(min(angle1, angle2)))
    
# TODO: optimization - better fit
def getApproxCenterLanes(left_lane_lines, right_lane_lines, lane_calibration):
    """
    Find approximate lane lines for center lane
    c1, c2, c3 are calibration points 
        c2 vanishing point
        c1 calibration triangle left point
        c3 calibration triangle right point
    """
    c1x, c1y, c2x, c2y, c3x, c3y = getLaneCalibrationPoints(lane_calibration)
    
    # Find approx left lane
    min_dist_val = None
    min_dist_index = None

    #left_lane_lines = sorted(left_lane_lines, key = sortLineFcn, reverse=True)
    for index,(x1,y1,x2,y2) in enumerate(left_lane_lines):
        d = linefn.segments_distance(x1,y1,x2,y2,c1x,c1y,c2x,c2y)
        if min_dist_val == None or d < min_dist_val:
            min_dist_index = index 
            min_dist_val = d
    center_lane_l = left_lane_lines[min_dist_index] if min_dist_index != None else None
    
    # Find approx right lane
    min_dist_val = None
    min_dist_index = None
    #right_lane_lines = sorted(right_lane_lines, key = sortLineFcn, reverse=True)
    for index,(x1,y1,x2,y2) in enumerate(right_lane_lines):
        d = linefn.segments_distance(x1,y1,x2,y2,c2x,c2y,c3x,c3y)
        if min_dist_val == None or d < min_dist_val:
            min_dist_index = index 
            min_dist_val = d
    center_lane_r = right_lane_lines[min_dist_index] if min_dist_index != None else None
    
    return (center_lane_l, center_lane_r)

def RetrieveWorldCoordinates(point):
    if DISPLAY_FULL_IMAGE:
        return (point[0], point[1]+LANE_ROI_START_Y)
    else:
        return (point[0], point[1])

        
def drawCalibration(img, lane_calibration):
    c1x, c1y, c2x, c2y, c3x, c3y = getLaneCalibrationPoints(lane_calibration)
    cv2.line(img,RetrieveWorldCoordinates(c1x,c1y),RetrieveWorldCoordinates(c2x,c2y),0,5) 
    cv2.line(img,RetrieveWorldCoordinates(c2x,c2y),RetrieveWorldCoordinates(c3x,c3y),0,5) 
    cv2.line(img,RetrieveWorldCoordinates(c3x,c3y),RetrieveWorldCoordinates(c1x,c1y),0,5) 
        
def DrawResults(out_img, left_lanes, right_lanes, pleft_lane, pright_lane, lane_calibration):
    
    # Draw calibration Triangle
    if DRAW_CALIBRATION:
        drawCalibration(out_img, lane_calibration)
                
    if DRAW_FILTERED_LANES: # draw all matching lanes
        for (x1,y1,x2,y2) in left_lanes:   
            cv2.line(out_img,RetrieveWorldCoordinates(x1,y1),RetrieveWorldCoordinates(x2,y2),0,2) 
        for (x1,y1,x2,y2) in right_lanes:   
            cv2.line(out_img,RetrieveWorldCoordinates(x1,y1),RetrieveWorldCoordinates(x2,y2),0,2)   
                  
    if pleft_lane:                 
        cv2.line(out_img,RetrieveWorldCoordinates(pleft_lane[:2]),RetrieveWorldCoordinates(pleft_lane[2:4]),(0,255,255),10) 
    if pright_lane:
        cv2.line(out_img,RetrieveWorldCoordinates(pright_lane[:2]),RetrieveWorldCoordinates(pright_lane[2:4]),(0,255,255),10)
                      
if __name__ == "__main__":
 
    cv2.namedWindow('Lane Markers')
    imgs = glob.glob("images/*.png")
    
    intercepts = []
    try:
        fsel = int(sys.argv[1])
    except IndexError:
        fsel = None
        
    fnum = 0
    for fname in imgs:
        fnum+=1
        if fsel != None and fnum != fsel:
            continue
    
        # Load image and prepare output image
        original_img = cv2.imread(fname)
        img = cv2.split(original_img)[0] # remove redundant channels
        (height, width) = img.shape[:2]
        
        # Preprocessing ----------------
        # Lane - ROI
        lane_roi = img[LANE_ROI_START_Y:height, :]
        
        # Prepare output image
        out_img1 = np.ones(lane_roi.shape, dtype="uint8") * 255
        out_img2 = out_img1.copy()
        
        # Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
        lane_roi = clahe.apply(lane_roi)
        
        # Noise elimination
        lane_roi_filtered = cv2.GaussianBlur(lane_roi,GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIGX, GAUSSIAN_KERNEL_SIGY) 
        
        # Edge detection -----------------
        lane_roi_edges = cv2.Canny(lane_roi_filtered,threshold1=CANNY_HYSTERESIS_MINVAL,threshold2=CANNY_HYSTERESIS_MAXVAL, L2gradient=CANNY_L2_GRADIENT, apertureSize=CANNY_APERTURE_SIZE) 
        
        # Car hood black out
        mask = np.zeros(lane_roi_edges.shape, dtype=np.uint8) 
        roi_corners = np.array([[ (0,0), (0, 1092-600), (90, 1065-600), (251, 1037-600), (381, 1029-600), (522, 1018-600), (824, 1009-600), (940 ,1014-600), (1087, 1030-600), (1206, 1053-600), (1322, 1116-600), (1435, height-600),(1477, height-600), (1598, 988-600), (width, 0)]], dtype=np.int32)
        cv2.fillPoly(mask, roi_corners, 255)
        lane_roi_edges = cv2.bitwise_and(lane_roi_edges, mask)
        lane_roi_edges_copy = lane_roi_edges.copy()
        
        # Hough Lines -----------------
        lines = cv2.HoughLinesP(lane_roi_edges_copy,rho=1,theta=1*np.pi/180,threshold=HOUGH_MIN_VOTES,minLineLength = HOUGH_MIN_LINE_LEN,maxLineGap = HOUGH_MAX_LINE_GAP) # Probabilistic hough
        if lines == None or len(lines) == 0:
            continue
            
        if DRAW_ALL_HOUGH_LINES:
            for x1,y1,x2,y2 in lines[0]:
                cv2.line(lane_roi,(x1,y1),(x2,y2),0,2)
    
        left_lane_lines, right_lane_lines = getLinesInCenterLaneProximity(lines[0], LANE_CALIBRATION)         
        pleft, pright = getApproxCenterLanes(left_lane_lines, right_lane_lines, LANE_CALIBRATION)
        
        # Draw Results
        if DISPLAY_FULL_IMAGE:
            DrawResults(original_img, left_lane_lines, right_lane_lines, pleft, pright, LANE_CALIBRATION)
        else:
            DrawResults(lane_roi, left_lane_lines, right_lane_lines, pleft, pright, LANE_CALIBRATION)   
                            
        # Sample intercept
        left_x = pleft[0] if pleft != None else None
        right_x = pright[2] if pright != None else None
        
        print left_x, right_x
    
        intercepts.append((os.path.basename(fname), left_x, right_x))
        
        # Show image
        if DISPLAY_FULL_IMAGE:
            cv2.imshow('Lane Markers', original_img)
        else:
            cv2.imshow('Lane Markers (ROI)', lane_roi)

        key = cv2.waitKey(0)
        if key == 27 or fsel != None:
            sys.exit(0)

    # CSV output
    with open('intercepts.csv', 'w') as f:
        writer = csv.writer(f)    
        writer.writerows(intercepts)
         
    cv2.destroyAllWindows();



"""
# Contours
TODO: shape matching and fitting 
(contours, _) = cv2.findContours(lane_roi_edges_copy,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#contours = sorted(contours, key = cv2.contourArea, reverse = True) #[:100]
#cv2.drawContours(out_img1, contours, -1, 0, -1)
for cnt in contours:
    
    rect = cv2.minAreaRect(cnt)
    
    #cv2.contourArea(cnt) >= 80  and (rect[1][1] >= MIN_LANE_LENGTH or rect[1][0] >= MIN_LANE_LENGTH)
    if cv2.isContourConvex(cnt) == False \
            and rect[0][0] >= 100 and rect[0][0] <= 1400 \
            and (rect[1][1] >= MIN_LANE_LENGTH or rect[1][0] >= MIN_LANE_LENGTH) \
            and ( rect[1][1]/float(max(rect[1][0], 0.001))  >= 5 or rect[1][0]/float(max(rect[1][1], 0.001)) >= 5 ) and abs(rect[2]) >= MIN_LANE_ANGLE: # and abs(rect[2]) <= MAX_LANE_ANGLE:
        # bounding box
        #box = cv2.cv.BoxPoints(rect)
        #box = np.int0(box)
        #cv2.drawContours(out_img1, [box], 0, 0, -1)
        #contour
        cv2.drawContours(out_img1, cnt, -1, 0, -1)
"""