from collections import namedtuple
from typing import List, Tuple, Iterator

import cv2
import numpy

Point = namedtuple('Point', 'x y')
Box = namedtuple('Box', 'x y w h')

MARKED_COLOR = [255, 0, 255]
LINE_COLOR = [70, 80, 192]
COLOR_THRESHOLD = 5
SCALE = 0.2

def is_line_pixel(pixel: numpy.ndarray) -> bool:
    return LINE_COLOR[0] - COLOR_THRESHOLD <= pixel[0] <= LINE_COLOR[0] + COLOR_THRESHOLD and LINE_COLOR[
        1] - COLOR_THRESHOLD <= pixel[1] <= LINE_COLOR[1] + COLOR_THRESHOLD and LINE_COLOR[2] - COLOR_THRESHOLD <= \
           pixel[2] <= LINE_COLOR[2] + COLOR_THRESHOLD


def inside(point: Point, box: Box) -> bool:
    return box.x <= point.x <= box.x + box.w and box.y <= point.y <= box.y + box.h


def expand_pixel(img: numpy.ndarray, pos: Point, candidates: List[Point]) -> ():
    img[pos.y, pos.x] = MARKED_COLOR

    # Check for black pixels which are node candidates
    if (img[pos.y - 1, pos.x] == [0, 0, 0]).all():
        candidates.append(Point(pos.y - 1, pos.x))
    if (img[pos.y, pos.x + 1] == [0, 0, 0]).all():
        candidates.append(Point(pos.y, pos.x + 1))
    if (img[pos.y, pos.x - 1] == [0, 0, 0]).all():
        candidates.append(Point(pos.y, pos.x - 1))
    if (img[pos.y + 1, pos.x] == [0, 0, 0]).all():
        candidates.append(Point(pos.y + 1, pos.x))

    if is_line_pixel(img[pos.y - 1, pos.x]):
        expand_pixel(img, Point(pos.x, pos.y - 1), candidates)
    if is_line_pixel(img[pos.y, pos.x + 1]):
        expand_pixel(img, Point(pos.x + 1, pos.y), candidates)
    if is_line_pixel(img[pos.y, pos.x - 1]):
        expand_pixel(img, Point(pos.x - 1, pos.y), candidates)
    if is_line_pixel(img[pos.y + 1, pos.x]):
        expand_pixel(img, Point(pos.x, pos.y + 1), candidates)


def get_adjacent_nodes(img: numpy.ndarray, start: Point, startbox: Box) -> Iterator[Tuple[int, int]]:
    # Downsample image to prevent stack overflow
    scale_img = cv2.resize(img,  # original image
                           (0, 0),  # set fx and fy, not the final size
                           fx=SCALE,
                           fy=SCALE,
                           interpolation=cv2.INTER_NEAREST)

    # Create empty candidates list to be filled by the recursive pixel filling algorithm
    candidates = []
    scale_start = Point(int(start.x * SCALE), int(start.y * SCALE))
    expand_pixel(scale_img, scale_start, candidates)
    cv2.imwrite('algo.png', scale_img)

    # Upscale node coordinates
    return map(lambda node: (int(node[0]/SCALE), int(node[1]/SCALE)), candidates)


img = cv2.imread('res.png')
nodeList = get_adjacent_nodes(img, Point(270, 665), (0, 0, 0, 0))

for node in nodeList:
    cv2.circle(img,(node[1], node[0]), 2, (255, 0, 0), 3)
    print(node)

cv2.imwrite('nodes.png', img)
