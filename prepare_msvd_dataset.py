# TODO setup a class for utilization of MSVD dataset. Take hints from prepare_mscoco_dataset.py. Make sure if the system has a backup, dataset should be saved to the backup. 

from pathlib import Path, PurePath
import pathlib
import requests
from tqdm import tqdm
from zipfile import ZipFile
from typing import List, Tuple
import json
import random

class MSVDDataset():

    def __init__(self) -> None:
        # name of the dataset.
        self.name = "MSVD"

        # url links for the dataset train images in zip format.
        self.video_path = "http://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar"

        # url links for the dataset annotations in zip format.
        self.annotations_path = "https://github.com/jazzsaxmafia/video_to_sequence/files/387979/video_corpus.csv.zip"

        # Project root Path
        self.root_folder = Path("/media/envisage/Yeni Birim/Selman/")

        # dataset Path
        self.dataset_folder = self.root_folder / "MSVD"

        # train images Path
        self.features_folder = self.dataset_folder / "features"

        # annotations Path
        self.annotations_folder = self.dataset_folder / "annotations"

        # train captions path
        self.captions = self.annotations_folder / "captions.json"

    def download_dataset(self) -> None:
        '''
            Download the dataset's train images, val images and their annotations into the MSCOCO folder.
        '''

        # dataset paths in a list.
        download_list = [self.video_path,self.annotations_path]

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

        # collect all the json files PosixPath into a list.
        list_of_annotations = list(self.annotations_folder.glob("*.json"))

        # Delete all the json's other than train and val captions.
        for ann in list_of_annotations:
            # Check if it is train captions or val captions if not proceed to deletion process.
            if ann != self.captions:
                # Check if the json file exists.
                if ann.exists():
                    # Delete the json file.
                    ann.unlink()


run = MSVDDataset()
run.download_dataset()