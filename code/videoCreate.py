# VIDEO_CREATE_PY
# 02.05.2023
# Ahmert
# selfie video creator

import os
import numpy as np
import math
import cv2
from datetime import datetime
import dlib
from PIL import Image
from pillow_heif import register_heif_opener

dataFolder = '../data'
blenderC = 8
freezerC = 3

def traverseFolder(path):
    photoList = []
    fileCount = 0
    
    with os.scandir(path) as it:
        for entry in it:
            if ((entry.name.lower().endswith('.jpeg')) or (entry.name.lower().endswith('.jpg'))) and entry.is_file():
                fileCount += 1
                print(str(entry.path) + ' <> ' + str(getDatePhotoTaken(entry.path)))
                photoList.append((entry.path, getDatePhotoTaken(entry.path)))
    
    print(str(fileCount) + ' photos were found')
    
    # sort
    photoList.sort(key=lambda tup: tup[1])
    return photoList
    
def getDatePhotoTaken(path):
    return Image.open(path)._getexif()[36867]

def rotateImage(image, center, angle):
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def rotatePoint(p, orig, angle):
    xN = ((p[0] - orig[0]) * math.cos(math.radians(angle))) - ((p[1] - orig[1]) * math.sin(math.radians(angle))) + orig[0];
    yN = ((p[0] - orig[0]) * math.sin(math.radians(angle))) + ((p[1] - orig[1]) * math.cos(math.radians(angle))) + orig[1];
    
    return [xN, yN]

def getLandmarks(imgPath, imgDate, detector, predictor):
    global leftEye
    global croppedImg
    global crops
    
    img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    
    print(img.shape)
    scaledH = 960
    scaledW = img.shape[1] / img.shape[0] * scaledH
    print(str(scaledW) + ' ' + str(scaledH))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    rect = rects[0]
    # get the landmark points
    shape = predictor(gray, rect)
	# convert it to NumPy Array
    shape_np = np.zeros((68, 2), dtype='int')
    for i in range(0, 68):
        shape_np[i] = (shape.part(i).x, shape.part(i).y)
    shape = shape_np

    '''
    # Display the landmarks
    for i, (x, y) in enumerate(shape):
	    # Draw the circle to mark the keypoint 
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    '''
    
    # calculate ref points
    leftEye = (shape[37] + shape[38] + shape[40] + shape[41]) / 4
    #cv2.circle(img, (int(leftEye[0]), int(leftEye[1])), 5, (0, 0, 255), -1)
    rightEye = (shape[43] + shape[44] + shape[46] + shape[47]) / 4
    #cv2.circle(img, (int(rightEye[0]), int(rightEye[1])), 5, (0, 0, 255), -1)
    chin = shape[8]
    
    # angle
    diff = rightEye - leftEye
    angle = math.degrees(math.atan2(diff[1], diff[0]))
    print(angle)
    
    leftBorder = int(leftEye[0] + 1.3 * (leftEye[0] - rightEye[0]))
    rightBorder = int(rightEye[0] + 1.3 * (rightEye[0] - leftEye[0]))
    midEye = (leftEye + rightEye) / 2
    '''
    topBorder = int(midEye[1] + 1.2 * (midEye[1] - chin[1]))
    bottomBorder = int(chin[1] + 0.5 *(chin[1] - midEye[1]))
    cv2.rectangle(img, (leftBorder, topBorder), (rightBorder, bottomBorder), (255, 255, 0), 4)
    '''
    
    # rotate
    rotatedImg = rotateImage(img, midEye, angle)
    chin = rotatePoint(chin, midEye, -angle)
    print(chin)
    
    nW = rightBorder - leftBorder
    nH = 4 * nW / 3
    #diffH = nH - chin[1] + midEye[1]
    #topBorder = int(midEye[1] - diffH * 1.2 / 1.7)
    topBorder = int(midEye[1] - nH * 1.2 / 2.7)
    bottomBorder = int(topBorder + nH)
    bottomAltBorder = int(chin[1] + nH * 0.5 / 2.7)
    
    print('cropped: ' + str(topBorder) + ':' + str(bottomBorder) + ' <> ' + str(leftBorder) + ':' + str(rightBorder))
    croppedImg = rotatedImg[topBorder:bottomBorder, leftBorder:rightBorder]
    croppedAltImg = rotatedImg[topBorder:bottomAltBorder, leftBorder:rightBorder]
    
    resizedImg = cv2.resize(croppedImg, (900, 1200), interpolation = cv2.INTER_AREA)
    resizedAltImg = cv2.resize(croppedAltImg, (900, 1200), interpolation = cv2.INTER_AREA)
    
    '''
    cv2.circle(rotatedImg, (int(midEye[0]), int(midEye[1])), 5, (0, 0, 255), -1)
    cv2.circle(rotatedImg, (int(chin[0]), int(chin[1])), 5, (0, 0, 255), -1)
    imgRs = cv2.resize(rotatedImg, (int(scaledW), int(scaledH)))  
    cv2.imshow('image', imgRs)
    
    imgCResized = cv2.resize(croppedImg, (700, 800))  
    cv2.imshow('imageCropped', imgCResized)
    
    cv2.waitKey(0)
    '''
    
    dateOfPhoto = datetime.strptime(imgDate, '%Y:%m:%d %H:%M:%S')
    fileName = os.path.basename(os.path.normpath(imgPath))
    dateNameIndex = 0
    while True:    
        nFilePath = imgPath[:-(len(fileName) + 1)] + '/cropped/c' \
            + str(dateOfPhoto.year) + '_' + str(dateOfPhoto.month) \
            + '_' + str(dateOfPhoto.day) + '_' + str(dateNameIndex) +'.jpg'
        if(nFilePath in crops):
            dateNameIndex += 1
        else:
            break
        
    dateNameIndex = 0
    while True:
        nAltFilePath = imgPath[:-(len(fileName) + 1)] + '/croppedAlt/c' \
            + str(dateOfPhoto.year) + '_' + str(dateOfPhoto.month) \
            + '_' + str(dateOfPhoto.day) + '_' + str(dateNameIndex) +'.jpg'
        if(nAltFilePath in crops):
            dateNameIndex += 1
        else:
            break
        
    print(nFilePath)
    cv2.imwrite(nFilePath, resizedImg)
    cv2.imwrite(nAltFilePath, resizedAltImg)
    
    return nFilePath

def main():
    global photos
    global crops
    
    register_heif_opener()
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('../model/shape_predictor_68_face_landmarks.dat')
    
    photos = traverseFolder(dataFolder)
    
    '''
    crops = []
    
    for photo in photos:
        cropFile = getLandmarks(photo[0], photo[1], detector, predictor)
        crops.append(cropFile)
        
    with open('croplist.txt', 'w') as filehandle:
        for listitem in crops:
            filehandle.write(f'{listitem}\n')
    '''
    
    #getLandmarks(photos[74][0], photos[74][1], detector, predictor)
    
    crops = []
    with open('croplist.txt', 'r') as filehandle:
        for line in filehandle:
            curr_place = line[:-1]
            crops.append(curr_place)
        
    # create video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter('out.avi', fourcc, 24, (900, 1200))
    
    prevImg = cv2.imread(crops[0])
    
    for crop in crops:
        img = cv2.imread(crop)
        
        for j in range(blenderC):
            blended = cv2.addWeighted(img, (j + 1) / blenderC, prevImg, (blenderC - j - 1) / blenderC, 0.0)
            video.write(blended)
                
        for i in range(freezerC):
            video.write(img)
        
        prevImg = img
        print(crop + ' frames were added.')
    
    for i in range(freezerC * 3):
        video.write(img)
    
    cv2.destroyAllWindows()
    video.release()

    print('terminated')

if __name__ == '__main__':
    main()
    
    