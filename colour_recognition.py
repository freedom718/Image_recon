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

Save_Data = True  # save the data in one file
Timing = False  # timing the operation
Step_color = False  # view the stages of colour recognition
Static_Test = False  # for distance test
Rover_Marker = False  # saves images into their individual marker files

# local path
script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in


def colour(counter, marker, distance):
  if Timing:
    Start_of_code = time.time()

  results = []  # empty list
  for k in range(1, counter + 1):
    num_storage = []  # store tuple colour's value in RGB
    name_storage = []  # stores the colour's name
    img = cv2.imread("colour%d.png" % k)

    height, width, numchannels = img.shape

    roi = img[int((height / 2) - (height / 2) * 0.85):int((height / 2) + (height / 2) * 0.85),
          int((width / 2) - (width / 2) * 0.85):int((width / 2) + (width / 2) * 0.85)]

    nroi = cv2.resize(roi, (100, 100))

    imgGray = cv2.cvtColor(nroi, cv2.COLOR_BGR2GRAY)

    # img = cv2.imread("colour%d.png" % k, cv2.COLOR_BGR2RGB)
    Gauss = cv2.GaussianBlur(imgGray, (5, 5), 0)
    # blur = cv2.medianBlur(Gauss, 7)
    # fliter = cv2.bilateralFilter(blur, 15, 75, 75)
    # kernel = np.ones((10, 10), np.uint8)
    # erode = cv2.erode(Gauss, kernel, iterations=10)
    # dilation = cv2.dilate(erode, kernel, iterations=20)
    # denoised = cv2.fastNlMeansDenoisingColored(dilation, None, 10, 10, 7, 21)

    ret, otsu = cv2.threshold(Gauss, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inpaint = cv2.inpaint(nroi, otsu, 3, cv2.INPAINT_NS)

    h = 8
    w = 8
    # h, w = img[:2]

    current_mode = 0

    # # defined boundaries HSV
    # boundaries = [("black", [0, 0, 0], [179, 255, 50]), ("white", [0, 0, 185], [179, 30, 255]),
    #                 ("orange", [10, 30, 50], [25, 255, 255]), ("yellow", [25, 30, 50], [35, 255, 255]),
    #                 ("green", [35, 30, 50], [85, 255, 255]), ("blue", [85, 30, 50], [130, 255, 255]),
    #                 ("purple", [130, 30, 50], [145, 255, 255]), ("pink", [145, 30, 50], [165, 255, 255]),
    #                 ("red", [165, 30, 50], [179, 255, 255]), ("red", [0, 30, 50], [10, 255, 255]),
    #                 ("grey", [0, 0, 50], [179, 30, 185])]

    # # defined boundaries RGB
    # boundaries = [("black", [0, 0, 0]), ("white", [255, 255, 255]),
    #               ("orange", [255, 165, 0]), ("yellow", [255, 255, 0]),
    #               ("green", [0, 128, 0]), ("blue", [0, 0, 255]),
    #               ("purple", [128, 0, 128]), ("pink", [255, 192, 203]),
    #               ("red", [255, 0, 0]), ("grey", [128, 128, 128]),
    #               ("aqua", [0, 255, 255]), ("fuchsia", [255, 0, 255]),
    #               ("silver", [192, 192, 192]), ("maroon", [128, 0, 0]),
    #               ("olive", [128, 128, 0]), ("lime", [0, 255, 0]),
    #               ("teal", [0, 128, 128]), ("navy", [0, 0, 128]),]

    # # defined boundaries RGB
    # boundaries = [("black", [0, 0, 0]), ("white", [255, 255, 255]),
    #               ("yellow", [255, 255, 0]), ("purple", [128, 0, 128]),
    #               ("green", [0, 128, 0]), ("blue", [0, 0, 255]),
    #               ("red", [255, 0, 0]), ("grey", [128, 128, 128]),
    #               ("blue", [0, 255, 255]), ("pink", [255, 0, 255]),
    #               ("grey", [192, 192, 192]), ("red", [128, 0, 0]),
    #               ("yellow", [128, 128, 0]), ("green", [0, 255, 0]),
    #               ("blue", [0, 128, 128]), ("blue", [0, 0, 128])]

    # # defined boundaries HSV
    # boundaries = [("black", [0, 0, 0]), ("white", [0, 0, 255]),
    #               ("yellow", [30, 255, 255]), ("purple", [150, 255, 127]),
    #               ("green", [60, 255, 127]), ("blue", [120, 255, 255]),
    #               ("red", [0, 255, 255]), ("grey", [0, 0, 127]),
    #               ("blue", [90, 255, 255]), ("pink", [150, 255, 255]),
    #               ("grey", [0, 0, 191]), ("red", [0, 255, 127]),
    #               ("yellow", [30, 255, 127]), ("green", [60, 255, 255]),
    #               ("blue", [90, 255, 127]), ("blue", [120, 255, 127])]

    # # defined boundaries HSV
    # boundaries = [("black", [0, 0, 0], [179, 255, 50]), ("white", [0, 0, 179], [179, 38, 255]),
    #               ("orange", [15, 38, 50], [22, 255, 255]), ("yellow", [23, 38, 50], [44, 255, 255]),
    #               ("yellow green", [45, 38, 50], [52, 255, 255]), ("green", [53, 38, 50], [74, 255, 255]),
    #               ("green cyan", [75, 38, 50], [82, 255, 255]), ("cyan", [83, 38, 50], [104, 255, 255]),
    #               ("blue cyan", [105, 38, 50], [112, 255, 255]), ("blue", [113, 38, 50], [134, 255, 255]),
    #               ("violet", [135, 38, 50], [142, 255, 255]), ("magenta", [143, 38, 50], [164, 255, 255]),
    #               ("red magenta", [165, 38, 50], [172, 255, 255]), ("red", [0, 38, 50], [14, 255, 255]),
    #               ("red", [173, 38, 50], [180, 255, 255]), ("gray", [0, 0, 50], [179, 38, 179])]

    # Colour Boundaries red, blue, yellow and gray HSV
    # this requires dtype="uint16" for lower and upper & HSV = np.float32(HSV) before the conversion of HSV_FULL
    boundaries = [("black", [0, 0, 0], [360, 255, 50]), ("white", [0, 0, 179], [360, 38, 255]),
                  ("orange", [15, 38, 50], [31, 255, 255]), ("yellow", [31, 38, 50], [75, 255, 255]),
                  ("yellow green", [75, 38, 50], [91, 255, 255]), ("green", [91, 38, 50], [135, 255, 255]),
                  ("green cyan", [135, 38, 50], [150, 255, 255]), ("cyan", [150, 38, 50], [195, 255, 255]),
                  ("blue cyan", [195, 38, 50], [210, 255, 255]), ("blue", [210, 38, 50], [255, 255, 255]),
                  ("violet", [255, 38, 50], [270, 255, 255]), ("magenta", [270, 38, 50], [315, 255, 255]),
                  ("red magenta", [315, 38, 50], [330, 255, 255]), ("red", [0, 38, 50], [15, 255, 255]),
                  ("red", [330, 38, 50], [360, 255, 255]), ("gray", [0, 0, 50], [360, 38, 179])]

    resizeBGR = cv2.resize(nroi, (w, h))  # reduces the size of the image so the process would run fast

    if Save_Data:
      if not os.path.exists("color"):
        os.makedirs("color")
      color_result = "color/{0}_{1}.png".format(marker, k)
      color_result_des = os.path.join(script_dir, color_result)
      cv2.imwrite(color_result_des, nroi)

    if Static_Test:
      color_result = "{0}/{1}_{2}.png".format(distance, marker, k)
      color_result_des = os.path.join(script_dir, color_result)
      cv2.imwrite(color_result_des, nroi)

    if Rover_Marker:
      marker_name = "marker={0}".format(marker)
      color_result = "{0}/{1}_{2}.png".format(marker_name, marker, k)
      color_result_des = os.path.join(script_dir, color_result)
      cv2.imwrite(color_result_des, nroi)

    # print(resizeBGR[1,1])

    resizeBGR = np.float32(resizeBGR)
    resizeHSV = cv2.cvtColor(resizeBGR, cv2.COLOR_BGR2HSV_FULL)
    resizeHSV[:, :, 1] = np.dot(resizeHSV[:, :, 1], 255)
    # resizeRGB = cv2.cvtColor(resizeBGR, cv2.COLOR_BGR2RGB)

    # print(resizeHSV[1,1])

    if Step_color:
      # for x in range(0, w):
      #   for y in range(0, h):
      #     num_storage.append(resizeRGB[x, y])

      # print(num_storage)

      cv2.imshow("fliter", dilation)
      cv2.imshow("denos", denoised)
      cv2.imshow("resize", resizeBGR)
      cv2.waitKey(0)
    # end if

    # for i in range(0, h):
    #   for j in range(0, w):
    #     RGB = resizeHSV[i, j]
    #     differences = []
    #     for (name, value) in boundaries:
    #       for component1, component2 in zip(RGB, value):
    #         difference = sum([abs(component1 - component2)])
    #         differences.append([difference, name])
    #     differences.sort()
    #     name_storage.append(differences[0][1])
    #
    # majority = Counter(name_storage)
    # results.append(majority.most_common(1)[0][0])

    # for i in range(0, h):
    #   for j in range(0, w):
    #     # RGB = []
    #     RGB = resizeRGB[i, j]
    #     # num_storage.append(RGB)
    #     # Finds the nearest colour name within the webcolors dataset by converting the classification to rgb then then find the closest the square is to remove the negative value.
    #     try:
    #       colorname = webcolors.rgb_to_name(RGB)
    #       name_storage.append(colorname)
    #
    #     except ValueError:
    #       min_colours = {}
    #       for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
    #         r_c, g_c, b_c = webcolors.hex_to_rgb(key)
    #         rd = (r_c - RGB[0]) ** 2
    #         gd = (g_c - RGB[1]) ** 2
    #         bd = (b_c - RGB[2]) ** 2
    #         min_colours[(rd + gd + bd)] = name
    #       name_storage.append(min_colours[min(min_colours.keys())])
    #
    # majority = Counter(name_storage)
    #
    # results.append(majority.most_common(1)[0][0])

    # comparing each pixel of the picture and append the colour name in to a list (BGR to RGB to get the name)
    for (color, lower, upper) in boundaries:
      lower = np.array(lower, dtype="uint16")
      upper = np.array(upper, dtype="uint16")

      mask = cv2.inRange(resizeHSV, lower, upper)

      ratio = np.round((cv2.countNonZero(mask) / (resizeHSV.size / 3))*100, 2)
      if ratio > current_mode:
        current_mode = ratio
        name_storage.append(color)
      else:
        pass
    results.append(name_storage[-1])

  mode = Counter(results)

  if Step_color:
    print(name_storage)
    print(majority)
    print(mode)

  if mode == Counter():
    colourname = "None"
  else:
    colourname = mode.most_common(1)[0][0]

  if Timing:
    Duration_of_color_recognition = time.time() - Start_of_code
    print("Duraction of Colour Recognition = {0}".format(Duration_of_color_recognition))


    # with open('Character Color.csv', 'a') as csvfile:  # for testing purposes
    #   filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    #   filewriter.writerow(
    #     [str(marker), str(Duration_of_color_recognition)])

  return colourname

