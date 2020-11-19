import cv2
import json
import os

cam_name = 'G2'
cam = "cam6"
vid_path = f'/home/costa/data/seq7_cut/seq7_{cam_name}.mp4'
folder = f"{cam_name}_images"
#path = '/home/costa/Desktop/res6.1.mp4'

video = cv2.VideoCapture(vid_path)
# total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
success,image = video.read()
count = 0

if not os.path.exists(folder):
  os.makedirs(folder)

while success:
  with open(f'output_files/subject1/landmarks/landmarks_sub1_frame_{count}.json') as file:
    data_subj1 = json.load(file)
  with open(f'output_files/subject2/landmarks/landmarks_sub2_frame_{count}.json') as file:
    data_subj2 = json.load(file)

  if cam in data_subj1:
    for landmark in data_subj1[cam]["landmarks"]:
      image = cv2.circle(image, (int(landmark[0]), int(landmark[1])), 15, (0, 0, 255), -1)

  if cam in data_subj2:
    for landmark in data_subj2[cam]["landmarks"]:
      image = cv2.circle(image, (int(landmark[0]), int(landmark[1])), 15, (0, 255, 0), -1)

  cv2.imwrite(f"{folder}/frame%d.jpg" % count, image)     # save frame as JPEG file
  success,image = video.read()
  if count % 1000 == 0:
    print('Read a new frame: ', count)
  count += 1
