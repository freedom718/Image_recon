from typing import List, Any
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
import multiprocessing as mp
from multiprocessing import Process
from multiprocessing import Manager

"The following code is used for detection of a square containing another square within in there is a colour and " \
"alphnumeric character which is needed to be recognition and provide the location of the GPS coordinates of the Targets"

Timing = True
Step_color = False
Save_Data = False

def colour(counter, marker, distance):
  if Timing:
    Start_of_code = time.time()

  results = []  # empty list
  for k in range(1, counter + 1):
    num_storage = []  # store tuple colour's value in RGB

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

    resizeBGR = cv2.resize(inpaint, (w, h))  # reduces the size of the image so the process would run fast

    if Save_Data:
      cv2.imwrite(
        'C:/Users/kevin/Desktop/PhD_Project_and_Research/Coding/Editions/method A/color/{0}_{1}.png'.format(marker, k),
        inpaint)

    # print(resizeBGR[1,1])

    # resizeHSV = cv2.cvtColor(resizeBGR, cv2.COLOR_BGR2HSV)
    resizeRGB = cv2.cvtColor(resizeBGR, cv2.COLOR_BGR2RGB)

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

    # the for i is the number of thread currently it is for 4 threads

    with Manager() as manager:
      name_storage = manager.list()  # stores the colour's name
      for i in range(0, h, 2):
        p = Process(target=get_Pixel_Color_Names, args=(i, resizeRGB, name_storage))
        p.start()
      for i in range(0, h, 2):
        p.join()

      majority = Counter(name_storage)

    # print(list(name_storage))

    # print("global name_storage results = {0}".format(name_storage))

    results.append(majority.most_common(1)[0][0])
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

    with open('Character Color.csv', 'a') as csvfile:  # for testing purposes
      filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
      filewriter.writerow(
        [str(marker), str(Duration_of_color_recognition)])

  return colourname


def get_Pixel_Color_Names(i, pixels, name_storage):
  print("I am thread {0}".format(i))

  w = 8
  for k in range(i, i + 1):
    for j in range(0, w):
      # RGB = []
      RGB = pixels[k, j]
      # num_storage.append(RGB)
      # Finds the nearest colour name within the webcolors dataset by converting the classification to rgb then then find the closest the square is to remove the negative value.
      try:
        colorname = webcolors.rgb_to_name(RGB)
        name_storage.append(colorname)

      except ValueError:
        min_colours = {}
        for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
          r_c, g_c, b_c = webcolors.hex_to_rgb(key)
          rd = (r_c - RGB[0]) ** 2
          gd = (g_c - RGB[1]) ** 2
          bd = (b_c - RGB[2]) ** 2
          min_colours[(rd + gd + bd)] = name
        name_storage.append(min_colours[min(min_colours.keys())])

  # print(name_storage)
