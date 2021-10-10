from pathlib import Path, PurePath
import pathlib
import requests
from tqdm import tqdm
from zipfile import ZipFile
from typing import List, Tuple
import json
import random
from torchvision.datasets.utils import _get_google_drive_file_id, _extract_zip, extract_archive

# Default number for seed is 0.
random.seed(0)

def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)  

def load_json_list(json_path: pathlib.Path, test: bool=False) -> Tuple[List[str], List[str]]:
    # Initialize ids list.
    video_ids_set = set()
    # Initialize video_paths list.
    video_paths = []
    # Initialize ids list.
    video_ids = []
    # Initialize train captions list.
    captions = []
    # start times
    start_times = []
    # end times
    end_times = []
    # Load json file in train_data
    data = json.loads(json_path.read_bytes())

    # Go through train data.
    print(f'Loading {str(json_path)} data...')
    previous_id_len = 0
    videos = data['videos']
    for annotation in tqdm(data['sentences']):
        # load caption and add start-of-caption and end-of-caption words.
        caption = 'boc ' + annotation['caption'] + ' eoc'
        # load id and add 0s till the id's string length is 12 which is the complete name of the video.
        video_id = annotation['video_id']
        # extract the video part in id. ex: video2869:str -> 2869:int
        id = int(video_id[5:])
        if test:
            video_info = videos[id-7010]
        else:
            video_info = videos[id]
            
        video_start_time = video_info['start time']
        video_end_time = video_info['end time']

        if test:
            video_ids_set.add(id)
            if len(video_ids_set) > previous_id_len:
                video_paths.append(video_id)
                video_ids.append(id)
                previous_id_len = len(video_ids)
                start_times.append(video_start_time)
                end_times.append(video_end_time)
                captions.append(caption)
        else:
            video_paths.append(video_id)
            video_ids.append(id)
            start_times.append(video_start_time)
            end_times.append(video_end_time)
            captions.append(caption)

    print('Data is loaded.')

    return video_paths, captions, video_ids, start_times, end_times

class MSRVTTDataset():
    '''
        Utilities for video captioning dataset of MSRVTT

        Initialize:
        # dt = MSRVTTDataset()

        Download MSRVTT Dataset:
        # dt.download_dataset()

        Load captions, paths, times and ids:
        # train_data, test_data = dt.load_data()

        # paths, captions, ids, start_times, end_times = zip(*train_data)
        In test data captions are not important therefore only one corresponding caption for a video added.
        # paths, captions, ids, start_times, end_times = zip(*test_data)

    '''
    def __init__(self, root_folder:pathlib.Path = None) -> None:

        # name of the dataset.
        self.name = "MSRVTT"

        # url links for the dataset train videos in zip format.
        self.train_path = ["https://drive.google.com/file/d/1XyZwkCGV2zF90jjfmkqVlWvWUVWqCr66", "train_val_videos.zip"]

        # url links for the dataset val videos in zip format.
        self.test_path = ["https://drive.google.com/file/d/17Q4Cq-QwO9ygjbVV9OBeJqGks2DwtltH", "test_videos.zip"]

        # url links for the dataset annotations in zip format.
        self.train_annotations_path = ["https://drive.google.com/file/d/1mglvNKhJ-igKiQFFk8RJfw9A8Xp1vLM6", "train_val_annotation.zip"]

        # url links for the dataset annotations in zip format.
        self.test_annotations_path = ["https://drive.google.com/file/d/16iaSq_qi3ve3coqZHokcWJCvIEv6AcH3", "test_annotation.zip"]

        # Project root Path
        self.root_folder = root_folder if root_folder is not None else Path('.')

        # Drive Path
        self.dataset_folder = self.root_folder / self.name

        # train videos Path
        self.train_folder = self.dataset_folder / "TrainValVideo"

        # val videos Path
        self.test_folder = self.dataset_folder / "TestVideo"

        # train videos Path
        self.train_features_folder = self.train_folder / "features_train"

        # val features Path
        self.val_features_folder = self.test_folder / "features_test"

        # train captions path
        self.train_annotations = self.dataset_folder / "train_val_videodatainfo.json"

        # val captions path
        self.test_annotations = self.dataset_folder / "test_videodatainfo.json"

    def download_dataset(self) -> None:
        '''
            Download the dataset's train videos, test videos and their annotations into the MSRVTT folder.
        '''

        # dataset paths in a list.
        download_list = [self.train_path, self.test_path, self.train_annotations_path, self.test_annotations_path]

        for path in download_list:
            # check if the dataset folder exists if not create it.
            if not self.dataset_folder.exists():
                Path(self.dataset_folder).mkdir(parents=True, exist_ok=True)
            
            # download_and_extract_archive(url=path, download_root=str(self.drive_data_zip_paths), extract_root=str(self.dataset_folder))
            file_id = _get_google_drive_file_id(path[0])
            file = self.dataset_folder / path[1]
            download_file_from_google_drive(id=file_id, destination=str(file))
            extract_archive(str(file), str(self.dataset_folder))
            file.unlink()
            
    def load_data(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        '''
            Load the MSCOCO captions and their corresponding video ids.
            paths, captions, ids, start_times, end_times
        '''

        train_paths, train_captions, train_ids, train_start_time, train_end_time = load_json_list(self.train_annotations)
        test_paths, test_captions, test_ids, test_start_time, test_end_time = load_json_list(self.test_annotations, test=True)
        
        train_data = zip(train_paths, train_captions, train_ids, train_start_time, train_end_time)
        test_data = zip(test_paths, test_captions, test_ids, test_start_time, test_end_time)

        return train_data, test_data

# dt = MSRVTTDataset()
# dt.download_dataset()
# train_data, test_data = dt.load_data()

