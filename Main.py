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
import csv
import os
import threading
import GPS
import colour_recognition
import character_recognition

"The following code is used for detection of a square containing another square within in there is a colour and " \
"alphnumeric character which is needed to be recognition and provide the location of the GPS coordinates of the Targets"


def solution(counter, marker, distance, Start_of_code):
  print("detection of marker", marker, "located")
  print(character_recognition.character(counter, marker, distance) + " is located for marker", marker)
  print(colour_recognition.colour(counter, marker, distance) + " is the colour of ground marker", marker)

  if Timing:
    end_of_code = time.time() - Start_of_code
    print("Whole code duration = {0}".format(end_of_code))

    with open('Distance.csv', 'a') as csvfile:  # for testing purposes
      filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
      filewriter.writerow(
        [str(marker), str(character_recognition.character(counter, marker, distance)),
         str(Colour.colour(counter, marker, distance)), str(distance), str(end_of_code)])

  # if Static_Test:
  with open('results.csv', 'a') as csvfile:  # for testing purposes
    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    filewriter.writerow([str(marker), str(character_recognition.character(counter, marker, distance)),
                         str(colour_recognition.colour(counter, marker, distance))])

  if GPS and air == 1:
    # Get middle position within lists outputted by detection function
    middle = int(len(positions) / 2)

    print(GPS.GPS(centres[middle], headings[middle], positions[middle][0], positions[middle][1], positions[middle][2],
                  height_of_target[middle]) + " latitude and longitude of", marker)
  elif GPS and air != 1:
    # Get middle position within lists outputted by detection function
    middle = int(len(positions) / 2)

    print(GPS.GPS(centres[middle], headings[middle], positions[middle][0], positions[middle][1], positions[middle][2],
                  height_of_target[middle]) + " latitude, longitude and altitidue of", marker)
  counter = 0
  marker = marker + 1

  return counter, marker


def detection():
  print('Starting detection')

  # Initialising variable
  counter = 0
  marker = 1
  positions = []
  headings = []
  centres = []
  height_of_target = []
  square = 2
  if Timing:
    Start_of_code = time.time()
  else:
    Start_of_code = 0

  # if Static_Test:
  cap = cv2.VideoCapture("TestData2.mp4")  # video use

  # cap = cv2.VideoCapture(1)
  # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960) #800
  # cap.set(3, 960) #800
  # cap.set(4, 540) #800
  # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
  # cap.set(cv2.CAP_PROP_FPS, 60)

  time.sleep(2)  # allows the camera to start-up
  print('Camera on')
  # Run detection when camera is turn on
  # while (cap.isOpened()): # for video use
  while True:
    # the camera will keep running even after the if statement so it can detect multiple ground marker
    if counter == 0 or start - end < 5:
      ret, frame = cap.read()

      # Gathering data from Pixhawk
      if GPS:
        position = vehicle.location.global_relative_frame
        heading = vehicle.heading
      # end if

      # starting the timer for the length of time it hasn't found a target
      start = time.time()

      # applying image processing

      frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
      avg_a = np.average(frame_lab[:, :, 1])
      avg_b = np.average(frame_lab[:, :, 2])
      frame_lab[:, :, 1] = frame_lab[:, :, 1] - ((avg_a - 128) * (frame_lab[:, :, 0] / 255.0) * 1.1)
      frame_lab[:, :, 2] = frame_lab[:, :, 2] - ((avg_b - 128) * (frame_lab[:, :, 0] / 255.0) * 1.1)
      whitebalance = cv2.cvtColor(frame_lab, cv2.COLOR_LAB2BGR)

      gray = cv2.cvtColor(whitebalance, cv2.COLOR_BGR2GRAY)  # converts to gray
      blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # blur the gray image for better edge detection
      edged = cv2.Canny(blurred, 14, 10)  # the lower the value the more detailed it would be
      edged_copy = edged.copy()

      # find contours in the thresholded image and initialize the
      (contours, _) = cv2.findContours(edged_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # grabs contours

      # outer square
      for c in contours:
        peri = cv2.arcLength(c, True)  # grabs the contours of each points to complete a shape
        # get the approx. points of the actual edges of the corners
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        cv2.drawContours(edged_copy, [approx], -1, (255, 0, 0), 3)
        if Step_detection:
          cv2.imshow("contours_approx", edged_copy)
          cv2.imshow("blurred", blurred)

        if 4 <= len(approx) <= 6:
          (x, y, w, h) = cv2.boundingRect(approx)  # gets the (x,y) of the top left of the square and the (w,h)
          aspectRatio = w / float(h)  # gets the aspect ratio of the width to height
          area = cv2.contourArea(c)  # grabs the area of the completed square
          hullArea = cv2.contourArea(cv2.convexHull(c))
          solidity = area / float(hullArea)
          keepDims = w > 25 and h > 25
          keepSolidity = solidity > 0.9  # to check if it's near to be an area of a square
          keepAspectRatio = 0.6 <= aspectRatio <= 1.4
          if keepDims and keepSolidity and keepAspectRatio:  # checks if the values are true

            # captures the region of interest with a 5 pixel lesser in all 2D directions
            roi = frame[y:y + h, x:x + w]

            height, width, numchannels = frame.shape

            centre_region = (x + w / 2, y + h / 2)
            if GPS:
              centre_target = (y + h / 2, x + w / 2)

            # grabs the angle for rotation to make the square level
            angle = cv2.minAreaRect(approx)[-1]  # -1 is the angle the rectangle is at

            if 0 == angle:
              angle = angle
            elif -45 > angle > 90:
              angle = -(90 + angle)
            elif -45 > angle:
              angle = 90 + angle
            else:
              angle = angle

            rotated = cv2.getRotationMatrix2D(tuple(centre_region), angle, 1.0)

            imgRotated = cv2.warpAffine(frame, rotated, (width, height))  # width and height was changed

            imgCropped = cv2.getRectSubPix(imgRotated, (w, h), tuple(centre_region))

            HSVCropp = cv2.cvtColor(imgCropped, cv2.COLOR_BGR2HSV)

            if square == 2:
              color = imgCropped[int((h / 2) - (h / 4)):int((h / 2) + (h / 4)),
                      int((w / 2) - (w / 4)):int((w / 2) + (w / 4))]
            else:
              color = imgCropped

            if Step_detection:
              cv2.imshow("rotated image", imgCropped)
              cv2.imshow("inner square", color)

              new = cv2.rectangle(frame,  # draw rectangle on original testing image
                                  (x, y),
                                  # upper left corner
                                  (x + w,
                                   y + h),
                                  # lower right corner
                                  (0, 0, 255),  # green
                                  3)
              cv2.imshow("frame block", new)
              # print(HSVCropp[int((h / 2) - (h * (6 / 10))), int((w / 2) - (w * (6 / 10)))])

            # # Convert the image to grayscale and turn to outline of  the letter
            # g_rotated = cv2.cvtColor(imgCropped, cv2.COLOR_BGR2GRAY)
            # b_rotated = cv2.GaussianBlur(g_rotated, (5, 5), 0)
            # e_rotated = cv2.Canny(b_rotated, 70, 20)
            #
            # # uses the outline to detect the corners for the cropping of the image
            # (contours, _) = cv2.findContours(e_rotated.copy(), cv2.RETR_LIST,
            #                                  cv2.CHAIN_APPROX_SIMPLE)
            #
            # # inner square detection
            # for cny in contours:
            #   perin = cv2.arcLength(cny, True)
            #   approxny = cv2.approxPolyDP(cny, 0.01 * perin, True)
            #   if 4 <= len(approxny) <= 6:
            #     (xx, yy), (ww, hh), angle = cv2.minAreaRect(approxny)
            #     aspectRatio = ww / float(hh)
            #     keepAspectRatio = 0.7 <= aspectRatio <= 1.3
            #     angle = cv2.minAreaRect(approxny)[-1]
            #     keep_angle = angle == 0, 90, 180, 270, 360
            #     if keepAspectRatio and keep_angle:
            #       (xxx, yyy, www, hhh) = cv2.boundingRect(approxny)
            #       color = imgCropped[yyy:yyy + hhh, xxx:xxx + www]

            # appends the data of the image to the list
            if GPS:
              positions.append([position.lat, position.lon, position.alt])
              headings.append(heading)
              centres.append(centre_target)
              height_of_target.append(h)

            if Static_Test:
              distance = input("Distance it was taken")
            #  start - end < 5
            if not Static_Test:
              distance = 1

            # time that the target has been last seen
            end = time.time()
            # time.sleep(0.5)

            # keep count of number of saved images
            counter = counter + 1
            cv2.imwrite("colour%d.png" % counter, color)

            if Save_Data:
              frame_resize = cv2.resize(frame, 720, 480)
              whitebalance_resize = cv2.resize(whitebalance, 720, 480)
              comparison = np.hstack((frame_resize, whitebalance_resize))

              # access local files
              if not os.path.exists("results"):
                os.makedirs("results")
              if not os.path.exists("frames"):
                os.makedirs("frames")

              colour_result = "results/{0}_{1}.png".format(marker, counter)
              comparison_result = "frames/orginal_vs_whitebalance{0}_{1}.png".format(marker, counter)
              colour_result_des = os.path.join(script_dir, colour_result)
              comparison_result_des = os.path.join(script_dir, comparison_result)

              cv2.imwrite(colour_result_des, color)
              cv2.imwrite(comparison_result_des, comparison)

            print("Detected and saved a target")

            if Static_Test:
              # testing purposes
              if not os.path.exists(distance):
                os.makedirs(distance)

              colour_result = "{0}/results{1}_{2}.png".format(distance, marker, counter)
              roi_result = "{0}/captured{1}_{2}.png".format(distance, marker, counter)
              frame_result = "{0}/orginal{1}_{2}.png".format(distance, marker, counter)
              colour_result_des = os.path.join(script_dir, colour_result)
              roi_result_des = os.path.join(script_dir, roi_result)
              frame_result_des = os.path.join(script_dir, frame_result)

              cv2.imwrite(colour_result_des, color)
              cv2.imwrite(roi_result_des, roi)
              cv2.imwrite(frame_result_des, frame)

            else:
              distance = 0

            if Rover_Marker:
              marker_name = "marker={0}".format(marker)
              if not os.path.exists("marker_name"):
                os.makedirs("marker_name")

              colour_result = "{0}/results{1}_{2}.png".format(marker_name, marker, counter)
              roi_result = "{0}/captured{1}_{2}.png".format(marker_name, marker, counter)
              frame_result = "{0}/orginal{1}_{2}.png".format(marker_name, marker, counter)
              colour_result_des = os.path.join(script_dir, colour_result)
              roi_result_des = os.path.join(script_dir, roi_result)
              frame_result_des = os.path.join(script_dir, frame_result)

              cv2.imwrite(colour_result_des, color)
              cv2.imwrite(roi_result_des, roi)
              cv2.imwrite(frame_result_des, frame)

            if Step_detection:
              cv2.imshow("captured image", roi)
              # cv2.imshow("cropped", imgCropped)
              cv2.waitKey(0)
            # end if
            if counter == 7:
              counter, marker = solution(counter, marker, distance, Start_of_code)
    else:
      counter, marker = solution(counter, marker, distance, Start_of_code)

    if Step_camera:
      cv2.imshow('frame', frame)
      cv2.imshow('white_balance', whitebalance)
      cv2.imshow('edge', edged)
      k = cv2.waitKey(5) & 0xFF
      if k == 27:
        break
    # end if

  cap.release()
  cv2.destroyAllWindows()


def main():
  if GPS:
    print('Connecting to drone...')

    # Connect to vehicle and print some info
    vehicle = connect('192.168.0.156:14550', wait_ready=True, baud=921600)

    print('Connected to drone')
    print('Autopilot Firmware version: %s' % vehicle.version)
    print('Global Location: %s' % vehicle.location.global_relative_frame)
  detection()


if __name__ == "__main__":
  # GPS
  GPS = False

  # Steps
  Step_camera = False # stages of camera activating
  Step_detection = False # stages of detection
  Static_Test = False # for distance test
  Timing = False # timing the operation
  Rover_Marker = False # saves images into their individual marker files

  # Saving Data into files
  Save_Data = True

  # local path
  script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

  main()
