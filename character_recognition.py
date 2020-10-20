from dronekit import *
from pymavlink import mavutil
import time
from math import *
import cv2
import numpy as np
import imutils.video
from collections import Counter
import webcolors
import operator
import matplotlib.pyplot as plt
import csv
import os
import threading

"The following code is used for detection of a square containing another square within in there is a colour and " \
"alphnumeric character which is needed to be recognition and provide the location of the GPS coordinates of the Targets"

Step_letter = False  # view the stages of character recognition
Timing = False  # timing the operation
Save_Data = True  # save the data
Static_Test = False  # for distance test
Rover_Marker = False  # saves images into their individual marker files

# Load Character Contour Area
MIN_CONTOUR_AREA = 100

# Load training and classification data
npaClassifications = np.loadtxt("classwithi.txt", np.float32)
npaFlattenedImages = np.loadtxt("flatwithi.txt", np.float32)

# reshape classifications array to 1D for k-nn
npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))

# local path
script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in


class ContourWithData():
  # member variables ############################################################################
  npaContour = None  # contour
  boundingRect = None  # bounding rect for contour
  intRectX = 0  # bounding rect top left corner x location
  intRectY = 0  # bounding rect top left corner y location
  intRectWidth = 0  # bounding rect width
  intRectHeight = 0  # bounding rect height
  fltArea = 0.0  # area of contour
  intCentreX = 0
  intCentreY = 0

  def calculateRectTopLeftPointAndWidthAndHeight(self):  # calculate bounding rect info
    [intX, intY, intWidth, intHeight] = self.boundingRect
    self.intRectX = intX
    self.intRectY = intY
    self.intCentreX = intX / 2
    self.intCentreY = intY / 2
    self.intRectWidth = intWidth
    self.intRectHeight = intHeight
    self.fltDiagonalSize = math.sqrt((self.intRectWidth ** 2) + (self.intRectHeight ** 2))

  def checkIfContourIsValid(self, height, width):  # this is oversimplified, for a production grade program
    MAX_CONTOUR_AREA = height * width * 0.9
    if MIN_CONTOUR_AREA < self.fltArea < MAX_CONTOUR_AREA:
      return True
    return False


def character(counter, marker, distance):
  print('Starting recognition thread')
  guesses = [0] * 35  # create a list of 35 lists

  if Timing:
    Start_of_code = time.time()

  for i in range(1, counter + 1):
    try:
      allContoursWithData = []  # declare empty lists
      validContoursWithData = []  # we will fill these shortly

      # set heights and width to be able to read the image when comparing to flatten images
      h = 30
      w = 30

      img = cv2.imread("colour%d.png" % i)

      height, width, numchannels = img.shape

      roi = img[int((height / 2) - (height / 2) * 0.85):int((height / 2) + (height / 2) * 0.85),
            int((width / 2) - (width / 2) * 0.85):int((width / 2) + (width / 2) * 0.85)]

      resize = cv2.resize(roi, (100, 100))

      # Convert the image to grayscale and turn to outline of the letter
      gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)

      if Step_letter:
        plt.hist(gray.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')  # calculating histogram
        plt.show()

      newheight, newwidth = gray.shape

      # imgMaxContrastGrayscale = maximizeContrast(gray)

      ###########

      gauss = cv2.GaussianBlur(gray, (5, 5), 0)
      # thresh = cv2.adaptiveThreshold(gauss, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0)
      # kernel = np.ones((4, 4), np.uint8)
      # # mask = cv2.inRange(gauss, 170, 255)
      # edged = cv2.Canny(gauss, 10, 30)  # the lower the value the more detailed it would be
      # dilate = cv2.dilate(edged, kernel, iterations=1)
      # kernel = np.ones((3, 3), np.uint8)
      # open = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, kernel, iterations=1)
      # close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel, iterations=3)
      # dilation = cv2.dilate(close, kernel, iterations=4)
      # kernel = np.ones((4, 4), np.uint8)
      # erode = cv2.erode(dilation, kernel, iterations=4)
      # open = cv2.morphologyEx(erode, cv2.MORPH_OPEN, kernel, iterations=1)
      # Removes the noises on the grayscale image
      # denoised = cv2.fastNlMeansDenoising(erode, None, 10, 7, 21)

      # mask = cv2.inRange(gray, 100, 255)  # works in lab, 100 at home,
      # # cv2.waitKey(0)

      _, otsu = cv2.threshold(gauss, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

      # imgBlurred = cv2.GaussianBlur(gray, (5, 5), 0)                    # blur

      if Step_letter:
        cv2.imshow("resivvve", resize)
        # cv2.imshow("mask", mask)
        # cv2.imshow("gg",denoised)
        # cv2.imshow("img",img)
        # cv2.imshow("gray",gray)
        # cv2.imshow("ed", edged)
        # cv2.imshow("dil", dilate)
        cv2.imshow("otsu", otsu)
        cv2.waitKey(0)

      # Fill in the letter to detect the letter easily
      # kernel = np.ones((4, 4), np.uint8)
      # closing = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
      # dilation = cv2.dilate(closing, kernel, iterations=1)

      knn = cv2.ml.KNearest_create()  # initalise the knn
      # joins the train data with the train_labels
      knn.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

      # # filter image from grayscale to black and white
      # imgThresh = cv2.adaptiveThreshold(gauss,  # input image
      #                                   255,  # make pixels that pass the threshold full white
      #                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
      #                                   # use gaussian rather than mean, seems to give better results
      #                                   cv2.THRESH_BINARY,
      #                                   # invert so foreground will be white, background will be black
      #                                   11,  # size of a pixel neighborhood used to calculate threshold value
      #                                   0)  # constant subtracted from the mean or weighted mean
      #
      # newkernal = np.ones((3, 3), np.uint8)
      # opening = cv2.morphologyEx(imgThresh, cv2.MORPH_OPEN, newkernal, iterations=1)
      # eroding = cv2.erode(opening, kernel, iterations=1)
      # dilating = cv2.dilate(eroding, kernel, iterations=1)

      imgThreshCopy = otsu.copy()  # make a copy of the thresh image, this in necessary b/c findContours modifies the image

      if Save_Data:
        if not os.path.exists("otsu"):
          os.makedirs("otsu")
        thresh_result = "otsu/{0}_{1}contour.png".format(marker, i)
        thresh_result_des = os.path.join(script_dir, thresh_result)
        cv2.imwrite(thresh_result_des, imgThreshCopy)

      if Static_Test:
        thresh_result = "{0}/{1}_{2}contour.png".format(distance, marker, i)
        thresh_result_des = os.path.join(script_dir, thresh_result)
        cv2.imwrite(thresh_result_des, imgThreshCopy)

      if Rover_Marker:
        marker_name = "marker={0}".format(marker)
        thresh_result = "{0}/{1}_{2}contour.png".format(marker_name, marker, i)
        thresh_result_des = os.path.join(script_dir, thresh_result)
        cv2.imwrite(thresh_result_des, imgThreshCopy)

      (npaContours, _) = cv2.findContours(imgThreshCopy,
                                          # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                          cv2.RETR_LIST,  # retrieve the outermost contours only
                                          cv2.CHAIN_APPROX_SIMPLE)  # compress horizontal, vertical, and diagonal segments and leave only their end points

      if Step_letter:
        cv2.imshow("npaContours", imgThreshCopy)
        cv2.imshow("planb", imgThresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

      for npaContour in npaContours:  # for each contour
        contourWithData = ContourWithData()  # instantiate a contour with data object
        contourWithData.npaContour = npaContour  # assign contour to contour with data
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)  # get the bounding rect
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()  # get bounding rect info
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)  # calculate the contour area
        allContoursWithData.append(contourWithData)  # add contour with data object to list of all contours with data
      # end for

      for contourWithData in allContoursWithData:  # for all contours
        if contourWithData.checkIfContourIsValid(newheight, newwidth):  # check if valid
          validContoursWithData.append(contourWithData)  # if so, append to valid contour list
        # end if
      # end for

      validContoursWithData.sort(key=operator.attrgetter("intRectX"))  # sort contours from left to right
      validContoursWithData = removeInnerOverlappingChars(validContoursWithData)  # removes overlapping letters

      for contourWithData in validContoursWithData:  # for each contour
        new = cv2.cvtColor(cv2.rectangle(roi,  # draw rectangle on original testing image
                                         (contourWithData.intRectX, contourWithData.intRectY),  # upper left corner
                                         (contourWithData.intRectX + contourWithData.intRectWidth,
                                          contourWithData.intRectY + contourWithData.intRectHeight),
                                         # lower right corner
                                         (0, 255, 0),  # green
                                         2), cv2.COLOR_BGR2GRAY)  # thickness

        imgROI = otsu[contourWithData.intRectY + 1: contourWithData.intRectY + contourWithData.intRectHeight - 1,
                 # crop char out of threshold image
                 contourWithData.intRectX + 1: contourWithData.intRectX + contourWithData.intRectWidth - 1]

        imgROIResized = cv2.resize(imgROI,
                                   (w, h))  # resize image, this will be more consistent for recognition and storage

        if Save_Data:
          if not os.path.exists("otsu"):
            os.makedirs("otsu")
          resize_thresh_result = "otsu/{0}_{1}chosen.png".format(marker, i)
          resize_thresh_result_des = os.path.join(script_dir, resize_thresh_result)
          cv2.imwrite(resize_thresh_result_des, imgROIResized)

        if Static_Test:
          resize_thresh_result = "{0}/{1}_{2}chosen.png".format(distance, marker, i)
          resize_thresh_result_des = os.path.join(script_dir, resize_thresh_result)
          cv2.imwrite(resize_thresh_result_des, imgROIResized)

        if Rover_Marker:
          marker_name = "marker={0}".format(marker)
          resize_thresh_result = "{0}/{1}_{2}chosen.png".format(marker_name, marker, i)
          resize_thresh_result_des = os.path.join(script_dir, resize_thresh_result)
          cv2.imwrite(resize_thresh_result_des, imgROIResized)

        # for i in range(0, 360, 90):
        #   angle = i
        #   rotate = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        #   imgROIResized = cv2.warpAffine(imgROIResized, rotate, (w, h))

        npaROIResized = imgROIResized.reshape((1, w * h))  # flatten image into 1d numpy array

        npaROIResized = np.float32(npaROIResized)  # convert from 1d numpy array of ints to 1d numpy array of floats

        if Step_letter:
          cv2.imshow("resize", imgROIResized)
          cv2.imshow("imgTestingNumbers", img)  # show input image with green boxes drawn around found digits
          cv2.waitKey(0)
        # end if

        # looks for the 3 nearest neighbours comparing to the flatten images (k = neighbours)
        retval, npaResults, neigh_resp, dists = knn.findNearest(npaROIResized, k=1)

        # current guess
        gg = int(npaResults[0][0])
        if Step_letter:
          print(gg)
        # Tranform guess in ASCII format into range 0-35
        if 49 <= gg <= 57:
          guesses[gg - 49] += 1
        elif 65 <= gg <= 90:
          guesses[gg - 56] += 1
    except:
      continue

  # find modal character guess
  # Initialise mode and prev variables for first loop through
  if Step_letter:
    print(guesses)
  mode = 0
  prev = guesses[0]
  for j in range(35):
    new = guesses[j]
    if new > prev:
      prev = guesses[j]
      mode = j
  # Transform back into ASCII
  if 0 <= mode <= 8:
    mode = mode + 49
  elif 9 <= mode <= 34:
    mode = mode + 56

  if Timing:
    Duration_of_character_recognition = time.time() - Start_of_code
    print("Duraction of Character Recognition = {0}".format(Duration_of_character_recognition))

    with open('Character_Duration.csv', 'a') as csvfile:  # for testing purposes
      filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
      filewriter.writerow(
        [str(marker), str(Duration_of_character_recognition)])

  return chr(mode)


def maximizeContrast(imgGrayscale):
  height, width = imgGrayscale.shape

  imgTopHat = np.zeros((height, width, 1), np.uint8)
  imgBlackHat = np.zeros((height, width, 1), np.uint8)

  structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

  imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
  imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

  imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
  imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

  return imgGrayscalePlusTopHatMinusBlackHat


def removeInnerOverlappingChars(listOfMatchingChars):
  # if we have two chars overlapping or to close to each other to possibly be separate chars, remove the inner (smaller) char,
  # this is to prevent including the same char twice if two contours are found for the same char,
  # for example for the letter 'O' both the inner ring and the outer ring may be found as contours, but we should only include the char once
  listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)  # this will be the return value

  for currentChar in listOfMatchingChars:
    for otherChar in listOfMatchingChars:
      if currentChar != otherChar:  # if current char and other char are not the same char . . .
        # if current char and other char have center points at almost the same location . . .
        if distanceBetweenChars(currentChar, otherChar) < (currentChar.fltDiagonalSize * 0.3):
          # if we get in here we have found overlapping chars
          # next we identify which char is smaller, then if that char was not already removed on a previous pass, remove it
          if currentChar.fltArea < otherChar.fltArea:  # if current char is smaller than other char
            if currentChar in listOfMatchingCharsWithInnerCharRemoved:  # if current char was not already removed on a previous pass . . .
              listOfMatchingCharsWithInnerCharRemoved.remove(currentChar)  # then remove current char
            # end if
          else:  # else if other char is smaller than current char
            if otherChar in listOfMatchingCharsWithInnerCharRemoved:  # if other char was not already removed on a previous pass . . .
              listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)  # then remove other char
            # end if
          # end if
        # end if
      # end if
    # end for
  # end for

  return listOfMatchingCharsWithInnerCharRemoved


def distanceBetweenChars(firstChar, secondChar):
  # use Pythagorean theorem to calculate distance between two chars
  intX = abs(firstChar.intCentreX - secondChar.intCentreX)
  intY = abs(firstChar.intCentreY - secondChar.intCentreY)

  return math.sqrt((intX ** 2) + (intY ** 2))
