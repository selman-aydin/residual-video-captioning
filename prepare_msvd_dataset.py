# TODO setup a class for utilization of MSVD dataset. Take hints from prepare_mscoco_dataset.py. Make sure if the system has a backup, dataset should be saved to the backup.

from pathlib import Path, PurePath
import pathlib
import requests
from tqdm import tqdm
from zipfile import ZipFile
from typing import List, Tuple
import json
import random
import pandas
import unicodedata
import re
import os
import csv
from shutil import copyfile


def load_json_list(json_path: pathlib.Path) -> Tuple[List[str], List[str], List[str]]:
    # Initialize video_paths list.
    video_paths = []
    # Initialize ids list.
    video_ids = []
    # Initialize train captions list.
    captions = []
    # Load json file in train_data
    data = json.loads(json_path.read_bytes())

    # Go through train data.
    print(f'Loading {str(json_path)} data...')
    for annotation in tqdm(data):
        # load caption and add start-of-caption and end-of-caption words.
        caption = 'boc ' + annotation['caption'] + ' eoc'

        # load video id
        video_id   = annotation['id']
        # load video path
        video_path = annotation['path']

        video_paths.append(video_path)
        video_ids.append(video_id)
        captions.append(caption)

    print('Data is loaded.')

    return video_paths, captions, video_ids

class MSVDDataset():

    def __init__(self) -> None:
        # name of the dataset.
        self.name = "MSVD"

        # url links for the dataset train images in zip format.
        self.video_path = "http://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar"

        # url links for the dataset annotations in zip format.
        self.annotations_path = "https://github.com/jazzsaxmafia/video_to_sequence/files/387979/video_corpus.csv.zip"

        # Project root Path
        self.root_folder = Path('/') / 'media' / 'envisage' / 'Yeni Birim' / 'Selman'

        # dataset Path
        self.dataset_folder = self.root_folder / "MSVD"

        # videos path
        self.videos_folder = self.dataset_folder / "YouTubeClips"

        # train videos Path
        self.train_folder = self.dataset_folder / "train"

        # test videos Path
        self.test_folder = self.dataset_folder / "test"

        # val videos Path
        self.val_folder = self.dataset_folder / "val"

        # train features Path
        self.train_features_folder = self.dataset_folder / "features_train"

        # test features Path
        self.test_features_folder = self.dataset_folder / "features_test"

        # val features Path
        self.val_features_folder = self.dataset_folder / "features_val"

        # annotations Path
        self.annotations_folder = self.dataset_folder / "annotations"

        # csv caption path
        self.captions = self.dataset_folder / "video_corpus.csv"

        # train captions path
        self.train_captions = self.annotations_folder / "captions_train.json"

        # val captions path
        self.test_captions = self.annotations_folder / "captions_test.json"

        # val captions path
        self.val_captions = self.annotations_folder / "captions_val.json"

    def download_dataset(self) -> None:
        '''
            Download the dataset's train images, val images and their annotations into the MSCOCO folder.
        '''

        # dataset paths in a list.
        download_list = [self.video_path, self.annotations_path]

        for path in download_list:
            r = requests.get(path, stream=True)
            # Get where to save the file i.e. MSCOCO/train2017.zip
            save_path = self.dataset_folder / PurePath(path).parts[-1]

            # check if the dataset folder exists if not create it.
            if not self.dataset_folder.exists():
                Path(self.dataset_folder).mkdir(parents=True, exist_ok=True)

            # Download data
            print("Downloading... " + path + " to " + str(save_path))
            with open(str(save_path), 'wb') as fd:
                for chunk in tqdm(r.iter_content(chunk_size=128)):
                    fd.write(chunk)

            # Extract all the contents of zip file in to the dataset folder
            with ZipFile(str(save_path), 'r') as zipObj:
                zipObj.extractall(path=str(self.dataset_folder))
            # delete the zip file.
            save_path.unlink()

    def splitVideos(self):
        """
            Splitting downloaded videos into train and val.
        """

        # check if the train and val folder exists if not create it.
        Path(self.train_folder).mkdir(parents=True, exist_ok=True)
        Path(self.test_folder).mkdir(parents=True, exist_ok=True)
        Path(self.val_folder).mkdir(parents=True, exist_ok=True)

        # get files in the videos_folder
        videoNames = self.videos_folder.glob("*.avi")
        # number of videos
        videoCount = len(list(videoNames))
        # calculate train and test video count
        trainSize = int( videoCount * 0.70)
        testSize  = trainSize + int(videoCount* 0.15)


        # copy videos to train and val folder
        for i, j in enumerate(self.videos_folder.glob('*.avi')):
            if i <= trainSize:
                videoId = j.stem + ".avi"
                original = self.videos_folder / videoId
                target = self.train_folder / videoId
                copyfile(original, target)

            elif trainSize< i <= testSize:
                videoId = j.stem + ".avi"
                original = self.videos_folder / videoId
                target = self.test_folder / videoId
                copyfile(original, target)

            else:
                videoId = j.stem + ".avi"
                original = self.videos_folder / videoId
                target = self.val_folder / videoId
                copyfile(original, target)

    def createJson(self):
        """
            Split csv file into train json and val json
        """

        #Collect train video id in list
        trainID = [i.stem for i in self.train_folder.glob("*.avi")]

        # Collect val video id in list
        testID = [i.stem for i in self.test_folder.glob("*.avi")]

        #Collect val video id in list
        valID =  [i.stem for i in self.val_folder.glob("*.avi")]


        json_train = []
        json_test = []
        json_val = []

        #read csv file
        with open(self.captions,encoding="utf8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:

                jsonElement = {"id": "", "path": "", "caption": ""}

                #if row is empty, pass
                if row == []:
                    pass

                #if caption is English
                elif row[6] == "English":

                    #if id in train
                    if str(row[0]) + "_" + str(row[1]) + "_" + str(row[2]) in trainID:
                        jsonElement["id"] = str(row[0]) + "_" + str(row[1]) + "_" + str(row[2])
                        jsonElement["path"] = str(row[0]) + "_" + str(row[1]) + "_" + str(row[2])
                        jsonElement["caption"] = str(row[7])
                        json_train.append(jsonElement)

                    # if id in test
                    elif str(row[0]) + "_" + str(row[1]) + "_" + str(row[2]) in testID:
                        jsonElement["path"] = str(row[0]) + "_" + str(row[1]) + "_" + str(row[2])
                        jsonElement["id"] = str(row[0]) + "_" + str(row[1]) + "_" + str(row[2])
                        jsonElement["caption"] = str(row[7])
                        json_test.append(jsonElement)

                    # if id in val
                    elif str(row[0]) + "_" + str(row[1]) + "_" + str(row[2]) in valID:
                        jsonElement["path"] = str(row[0]) + "_" + str(row[1]) + "_" + str(row[2])
                        jsonElement["id"] = str(row[0]) + "_" + str(row[1]) + "_" + str(row[2])
                        jsonElement["caption"] = str(row[7])
                        json_val.append(jsonElement)




        #check if the annotations folder exists if not create it.
        Path(self.annotations_folder).mkdir(parents=True, exist_ok=True)

        #save train id-caption list to json
        with open(self.train_captions, 'w') as f:
            json.dump(json_train, f)

        # save test id-caption list to json
        with open(self.test_captions, 'w') as f:
            json.dump(json_test, f)

        # save val id-caption list to json
        with open(self.val_captions, 'w') as f:
            json.dump(json_val, f)

    def load_data(self) -> Tuple[List[str], List[str], List[str]]:
        '''
            Load the MSCOCO captions and their corresponding video ids.
            paths, captions, ids, start_times, end_times
        '''

        train_paths, train_captions, train_ids = load_json_list(self.train_captions)
        train_data = zip(train_paths, train_captions, train_ids)

        val_paths, val_captions, val_ids = load_json_list(self.val_captions)
        val_data = zip(val_paths, val_captions, val_ids)

        test_paths, test_captions, test_ids = load_json_list(self.test_captions)
        test_data = zip(test_paths, test_captions, test_ids)

        return train_data,val_paths,test_paths

    def run(self):

        #self.downlocapad_dataset()
        #self.splitVideos()
        self.createJson()
        self.load_data()

a = MSVDDataset()
a.run()







