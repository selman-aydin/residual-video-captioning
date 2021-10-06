# TODO setup a class for utilization of MSVD dataset. Take hints from prepare_mscoco_dataset.py. Make sure if the system has a backup, dataset should be saved to the backup. 

from pathlib import Path, PurePath
import pathlib
import requests
from tqdm import tqdm
from zipfile import ZipFile
from typing import List, Tuple
import json
import random
from shutil import rmtree
import csv
from socket import gethostname
hostname = gethostname()

from torchvision.datasets.utils import download_and_extract_archive, download_url, _extract_zip

class MSVDDataset():

    def __init__(self, root_folder:pathlib.Path = None) -> None:
        # name of the dataset.
        self.name = "MSVD"

        # url links for the dataset train images in zip format.
        self.video_path = "http://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar"

        # url links for the dataset annotations in zip format.
        self.annotations_path = "https://github.com/jazzsaxmafia/video_to_sequence/files/387979/video_corpus.csv.zip"

        # Project root Path        
        self.root_folder = root_folder if root_folder is not None else Path('/') / 'media' / 'envisage' / 'Yeni Birim' / 'Selman' 

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

        # download videos
        download_and_extract_archive(url=self.video_path, download_root=str(self.dataset_folder), remove_finished=True)
        filename = Path(self.video_path).name
        archive = self.dataset_folder / filename
        archive.unlink()

        # check if the dataset folder exists if not create it.
        if not self.dataset_folder.exists():
            Path(self.dataset_folder).mkdir(parents=True, exist_ok=True)
        
        #download annotations
        filename = Path(self.annotations_path).name
        download_url(url=self.annotations_path, root=str(self.dataset_folder), filename=filename, md5=None)
        archive = self.dataset_folder / filename
        aux_dir = self.dataset_folder / '__MACOSX'
        _extract_zip(str(archive), str(archive.parent), None)
        archive.unlink()
        rmtree(aux_dir)

       



if 'ozkan' in hostname:
    dt = MSVDDataset(Path('.'))
else:
    dt = MSVDDataset()
dt.download_dataset()