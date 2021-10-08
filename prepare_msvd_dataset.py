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
from shutil import rmtree
from socket import gethostname
hostname = gethostname()

from torchvision.datasets.utils import download_and_extract_archive, download_url, _extract_zip


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

        # val videos Path
        self.val_folder = self.dataset_folder / "val"

        # train features Path
        self.train_features_folder = self.dataset_folder / "features_train"

        # val features Path
        self.val_features_folder = self.dataset_folder / "features_val"

        # annotations Path
        self.annotations_folder = self.dataset_folder / "annotations"

        # csv caption path
        self.captions = self.dataset_folder / "video_corpus.csv"

        # train captions path
        self.train_captions = self.annotations_folder / "captions_train.json"

        # val captions path
        self.val_captions = self.annotations_folder / "captions_val.json"

    def download_dataset(self) -> None:
        '''
            Download the dataset's train images, val images and their annotations into the MSCOCO folder.
        '''

        # download videos
        download_and_extract_archive(url=self.video_path, download_root=str(self.dataset_folder), remove_finished=True)
        filename = Path(self.video_path).name
        archive = self.dataset_folder / filename
        archive.unlink()

        # check if the dataset folder exists if not create it.
        if not self.dataset_folder.exists():
            Path(self.dataset_folder).mkdir(parents=True, exist_ok=True)

        # download annotations
        filename = Path(self.annotations_path).name
        download_url(url=self.annotations_path, root=str(self.dataset_folder), filename=filename, md5=None)
        archive = self.dataset_folder / filename
        aux_dir = self.dataset_folder / '__MACOSX'
        _extract_zip(str(archive), str(archive.parent), None)
        archive.unlink()
        rmtree(aux_dir)

    def splitVideos(self):
        """
            Splitting downloaded videos into train and val.
        """

        # check if the train and val folder exists if not create it.
        Path(self.train_folder).mkdir(parents=True, exist_ok=True)
        Path(self.val_folder).mkdir(parents=True, exist_ok=True)

        # get files in the videos_folder
        videoNames = self.videos_folder.glob("*.avi")

        # calculate train video size
        trainSize = int(len(list(videoNames)) * 0.80)

        # copy videos to train and val folder
        for i, j in enumerate(self.videos_folder.glob('*.avi')):
            if i < trainSize:
                videoId = j.stem + ".avi"
                original = self.videos_folder / videoId
                target = self.train_folder / videoId
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

        #Collect val video id in list
        valID =  [i.stem for i in self.val_folder.glob("*.avi")]


        json_train = []
        json_val = []

        #read csv file
        with open(self.captions,encoding="utf8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:

                jsonElement = {"id": "", "caption": ""}

                #if row is empty, pass
                if row == []:
                    pass

                #if caption is English
                elif row[6] == "English":

                    #if id in train
                    if str(row[0]) + "_" + str(row[1]) + "_" + str(row[2]) in trainID:

                        jsonElement["id"] = str(row[0]) + "_" + str(row[1]) + "_" + str(row[2])
                        jsonElement["caption"] = str(row[7])
                        json_train.append(jsonElement)

                    # if id in val
                    elif str(row[0]) + "_" + str(row[1]) + "_" + str(row[2]) in valID:
                        jsonElement["id"] = str(row[0]) + "_" + str(row[1]) + "_" + str(row[2])
                        jsonElement["caption"] = str(row[7])
                        json_val.append(jsonElement)

        #check if the annotations folder exists if not create it.
        Path(self.annotations_folder).mkdir(parents=True, exist_ok=True)

        #save train id-caption list to json
        with open(self.train_captions, 'w') as f:
            json.dump(json_train, f)

        # save val id-caption list to json
        with open(self.val_captions, 'w') as f:
            json.dump(json_val, f)

    def run(self):
        self.download_dataset()
        self.splitVideos()
        self.createJson()



if 'ozkan' in hostname:
    dt = MSVDDataset(Path('.'))
    dt.run()
else:
    dt = MSVDDataset()
    dt.run()



