import cv2
import numpy as np
import glob

img_array = []
filenames = glob.glob('/home/costa/src/DataPreprocessing/images/*.png')
frame_nbr = len(filenames)
img_path = '/home/costa/src/DataPreprocessing/images/'
img_prefix = 'img_step_'
for frame in range(frame_nbr):
    img = cv2.imread(f'{img_path}{img_prefix}{frame}.png')
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 60, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()