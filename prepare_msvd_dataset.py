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
import shutil
from shutil import rmtree
import csv
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

        #videos path
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

        #csv caption path
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



    def splitVıdeos(self):
        Path(self.train_folder).mkdir(parents=True, exist_ok=True)
        Path(self.val_folder).mkdir(parents=True, exist_ok=True)

        videoNames = []
        for root,directoires,files in os.walk( self.videos_folder):
            for filename in files:
                file = os.path.basename(filename).split(".",1)[0]
                videoNames.append(file)
        print(videoNames[0:5])

        trainSize = len(videoNames)*0.80
        valSize = len(videoNames)*0.20

        trainVideoID = []
        valVideoId = []

        for i in range(len(videoNames)):
            if i<trainSize:
                videoId = str(videoNames[i]) + ".avi"
                original =  self.videos_folder / videoId
                target = self.train_folder / videoId
                shutil.copyfile(original,target)
                trainVideoID.append(videoNames[i])
            else:
                videoId = str(videoNames[i]) + ".avi"
                original = self.videos_folder / videoId
                target = self.val_folder /videoId
                shutil.copyfile(original, target)
                valVideoId.append(videoNames[i])



    def createJson(self):

        trainID = []
        valID = []

        for root, directoires, files in os.walk(self.train_folder):
            for filename in files:
                file = os.path.basename(filename).split(".", 1)[0]
                trainID.append(file)

        for root, directoires, files in os.walk(self.val_folder):
            for filename in files:
                file = os.path.basename(filename).split(".", 1)[0]
                valID.append(file)

        csvCorpus = pandas.read_csv(self.captions)

        json_train = []
        json_val = []

        for i in range(len(csvCorpus.Language)):
            jsonElement = {"id": "","caption": ""}

            if csvCorpus.Language[i] == "English":

                if str(csvCorpus.VideoID[i]) + "_" + str(csvCorpus.Start[i]) + "_" + str(
                        csvCorpus.End[i]) in trainID:
                    jsonElement["id"] = str(csvCorpus.VideoID[i]) + "_" + str(csvCorpus.Start[i]) + "_" + str(
                        csvCorpus.End[i])
                    jsonElement["caption"] = str(csvCorpus.Description[i])
                    json_train.append(jsonElement)

                elif str(csvCorpus.VideoID[i]) + "_" + str(csvCorpus.Start[i]) + "_" + str(
                        csvCorpus.End[i]) in valID:
                    jsonElement["id"] = str(csvCorpus.VideoID[i]) + "_" + str(csvCorpus.Start[i]) + "_" + str(
                        csvCorpus.End[i])
                    jsonElement["caption"] = str(csvCorpus.Description[i])
                    json_val.append(jsonElement)

                else:
                    pass

        Path(self.annotations_folder).mkdir(parents=True, exist_ok=True)

        with open(self.train_captions, 'w') as f:
            json.dump(json_train, f)

        with open(self.val_captions, 'w') as f:
            json.dump(json_val, f)




if 'ozkan' in hostname:
    dt = MSVDDataset(Path('.'))
else:
    dt = MSVDDataset()
dt.download_dataset()