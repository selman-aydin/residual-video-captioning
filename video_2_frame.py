from prepare_msvd_dataset import MSVDDataset
import cv2
from pathlib import Path
import json


def calculateHistogram(img):

  # find frequency of pixels in range 0-255
  histr = cv2.calcHist([img], [0], None, [256], [0, 256])
  return histr

def video2Frame(inputFolder):
  json_histogram = []

  ids = list(inputFolder.glob('*.avi'))  # list of video files from input folder

  for id_path in ids:
    video_id = str(id_path).split("/")[-1].split(".")[0]
    vidcap = cv2.VideoCapture(str(id_path))
    success, image = vidcap.read()
    count = 0
    while success:
      jsonElement = {"id": "", "histogram": []}
      #cv2.imwrite(str(outputFolder)+"/"+video_id+"_%d.jpg" % count, image)  # save frame as JPEG file
      jsonElement["id"] = video_id + "_" + str(count)
      jsonElement["histogram"] = calculateHistogram(image)
      json_histogram.append(jsonElement)
      success, image = vidcap.read()
      print('Read a new frame: ', success)
      count += 1


  if str(inputFolder).split("/")[-1] == "train":
    with open(msvd_dt.dataset_folder/"train_histogram", 'w') as f:
      json.dump(str(json_histogram), f)

  elif str(inputFolder).split("/")[-1] == "test":
    with open(msvd_dt.dataset_folder/"test_histogram", 'w') as f:
      json.dump(str(json_histogram), f)

  else:
    with open(msvd_dt.dataset_folder/"val_histogram", 'w') as f:
      json.dump(str(json_histogram), f)




msvd_dt = MSVDDataset()


input_folders = [msvd_dt.train_folder, msvd_dt.val_folder, msvd_dt.test_folder]


for inp in input_folders:
  video2Frame(inp)





