import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import glob, os
import math

black_out_nodes = False # Set this to true to black out the found nodes
add_legend = True # Set this to true to label the found nodes
should_generate_labels = True # Set this to true to label an unlabeled image
should_generate_masks = False # Set this to true to generate mask images
image_to_label = 'ex4.png'

nodes = []

class node():
  def __init__(self, templateName, nodeDims, nodePosition):
    self.type = templateName
    self.nodePosition = nodePosition
    self.nodeDims = nodeDims

# Checks if a node (or rather the bounding box that contains the node)
# contains a given X-Y coordinate.
def nodeContainsCoordinate(x, y, node):
  topX = node.nodePosition[0]
  topY = node.nodePosition[1]
  botX = node.nodeDims[0] + topX
  botY = node.nodeDims[1] + topY
  return (topX <= x <= botX) and (topY <= y <= botY)

# Finds the node that contains a given X-Y coordinate. If it can't find
# such a node, it resurns None
def findContainingNode(x, y):
  for node in nodes:
    if nodeContainsCoordinate(x, y, node):
      return node
  return None

def posDist(a, b):
  return math.sqrt(pow((a.nodePosition[0]-b.nodePosition[0]), 2) + pow((a.nodePosition[1]-b.nodePosition[1]), 2))

# Finds matches for a given template in a file                       
# Params:
# - img_gray:       Image to look at in greyscale
# - img_rgb:        Image in colors, here we write the bounding boxes
# - templateImage:  The template we want to find
# - templateName:   The name of the template (for labeling)
# - threshold:      The threshold to apply for this matching
def matchImage(img_gray, img_rgb, templateImage, templateName, threshold):
  w, h = templateImage.shape[::-1]
  res = cv.matchTemplate(img_gray, templateImage, cv.TM_CCOEFF_NORMED)
  loc = np.where( res >= threshold )
  for pt in zip(*loc[::-1]):
    newPoint = node(templateName, [w, h], [pt[0], pt[1]])
    canAdd = True
    for oldPoint in nodes:
      if posDist(newPoint, oldPoint) < 80:
        canAdd = False
        break
    if canAdd:
      nodes.append(newPoint)
      print("Discovered template " + templateName + " in image at x =" + str(pt[0]) + ", y = " + str(pt[1]))
      if len(nodes) == 1:
        print("----- Appending, now has 1 entry")
      else:
        print("----- Appending, now has " + str(len(nodes)) + ' entries')
      if black_out_nodes:
        cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 0), -1)
      else:
        cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 0), 2)
      if add_legend:
        cv.putText(img_rgb, templateName, (pt[0] + 20, pt[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

# Searches for a template in an image. Depending on the rotMode param,
# it also searches for rotated versions of the template
# Params:
# - img_gray:       Image to look at in greyscale
# - img_rgb:        Image in colors, here we write the bounding boxes
# - templateName:   The name of the template (for loading the image)
# - threshold:      The threshold to apply for this matching
# - rotMode:        0 = Template is rotation invariant, need no rotation
#                   1 = Need to rotate template 90 degrees and search again
#                   2 = Need to rotate templat 90, 180 and 270 degrees and search again
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

# Generates labels & bounding boxes for one image
# Params:
# input_image:  Filename of the image to label
def generate_labels(input_image):
  img_rgb = cv.imread(input_image)
  img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

  for file in glob.glob("symbols/high_t/*.png"):
    match(img_gray, img_rgb, file, 0.8, 2)

  for file in glob.glob("symbols/low_t/*.png"):
    match(img_gray, img_rgb, file, 0.8, 2)

  for file in glob.glob("symbols/low_t/rot_inv/*.png"):
    match(img_gray, img_rgb, file, 0.8, 0)

  for file in glob.glob("symbols/low_t/rot_once/*.png"):
    match(img_gray, img_rgb, file, 0.8, 1)

  return img_rgb

# Main Method
if __name__ == "__main__":
  if should_generate_labels:
    res = generate_labels(image_to_label)
    cv.imwrite('res.png', res)

  if should_generate_masks:
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
