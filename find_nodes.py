import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import glob, os
import math

black_out_nodes = True # Set this to true to black out the found nodes
add_legend = False # Set this to true to label the found nodes
generate_labels = False # Set this to true to label an unlabeled image


def posDist(a, b):
  return math.sqrt(pow((a[0]-b[0]), 2) + pow((a[1]-b[1]), 2))

def matchImage(img_gray, img_rgb, templateImage, templateName, threshold):
  w, h = templateImage.shape[::-1]
  res = cv.matchTemplate(img_gray, templateImage, cv.TM_CCOEFF_NORMED)
  loc = np.where( res >= threshold )
  vectors = []
  for pt in zip(*loc[::-1]):
    newPoint = [pt[0], pt[1]]
    canAdd = True
    for oldPoint in vectors:
      if posDist(newPoint, oldPoint) < 20:
        canAdd = False
        break
    if canAdd:
      vectors.append(newPoint)
      print("Discovered template " + templateName + " in image at x =" + str(pt[0]) + ", y = " + str(pt[1]))
      print("----- Appending, vectors is now: " + str(vectors))
      if black_out_nodes:
        cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 0), -1)
      else:
        cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 0), 2)
      if add_legend:
        cv.putText(img_rgb, templateName, (pt[0] + 20, pt[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

def match(img_gray, img_rgb, templateName, threshold, rotMode):
  print("####################### LOOKING FOR " + templateName + " #######################")
  template = cv.imread(templateName, 0)

  rot90 = cv.rotate(template, cv.ROTATE_90_CLOCKWISE)
  rot180 = cv.rotate(rot90, cv.ROTATE_90_CLOCKWISE)
  rot270 = cv.rotate(rot180, cv.ROTATE_90_CLOCKWISE)

  print("Normal")
  matchImage(img_gray, img_rgb, template, templateName, threshold)

  if rotMode == 1 or rotMode == 2:
    print("90 deg")
    matchImage(img_gray, img_rgb, rot90, templateName, threshold)
    if rotMode == 2:
      print("180 deg")
      matchImage(img_gray, img_rgb, rot180, templateName, threshold)
      print("270 deg")
      matchImage(img_gray, img_rgb, rot270, templateName, threshold)

if generate_labels:
  img_rgb = cv.imread('ex4.png')
  img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

  os.chdir("symbols")
  for file in glob.glob("high_t/*.png"):
    match(img_gray, img_rgb, file, 0.88, 2)

  for file in glob.glob("low_t/*.png"):
    match(img_gray, img_rgb, file, 0.8, 2)

  for file in glob.glob("low_t/rot_inv/*.png"):
    match(img_gray, img_rgb, file, 0.8, 0)

  for file in glob.glob("low_t/rot_once/*.png"):
    match(img_gray, img_rgb, file, 0.8, 1)


  #match(img_gray, img_rgb, "lines/Crossing.png", (r, g, b), 0.8, 1)
  #match(img_gray, img_rgb, "lines/Junction.png", (r, g, b), 0.82, 2)

  cv.imwrite('../res.png', img_rgb)

black = (0, 0, 0)
grey = (100, 100, 100)
white = (255, 255, 255)

res_img = cv.imread('res.png')
hsv_img = cv.cvtColor(res_img, cv.COLOR_RGB2HSV)
line_mask = cv.inRange(hsv_img, grey, white)
node_mask = cv.inRange(hsv_img, black, grey)

kernel = np.ones((10, 10), np.uint8)
dilated_line_mask = cv.dilate(line_mask, kernel, iterations=1)

meeting_points = (node_mask & dilated_line_mask)

cv.imwrite('line_mask.png', line_mask)
cv.imwrite('node_mask.png', node_mask)
cv.imwrite('line_mask_dilated.png', dilated_line_mask)
cv.imwrite('meeting_points.png', meeting_points)
