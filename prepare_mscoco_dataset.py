from pathlib import Path, PurePath
import pathlib
import requests
from tqdm import tqdm
from zipfile import ZipFile
from typing import List, Tuple
import json
import random

# Default number for seed is 0.
random.seed(0)

def load_json_list(json_path: pathlib.Path, Val: bool=False) -> Tuple[List[str], List[str]]:
    # Initialize ids list.
    image_ids_set = set()
    # Initialize image_paths list.
    image_paths = []
    # Initialize ids list.
    image_ids = []
    # Initialize train captions list.
    captions = []
    # Load json file in train_data
    data = json.loads(json_path.read_bytes())

    # Go through train data.
    print(f'Loading {str(json_path)} data...')
    previous_id_len = 0
    for annotation in tqdm(data['annotations']):
        # load caption and add start-of-caption and end-of-caption words.
        caption = 'boc ' + annotation['caption'] + ' eoc'
        # load id and add 0s till the id's string length is 12 which is the complete name of the image.
        id = annotation['image_id']
        path = '%012d' % id

        if Val:
            image_ids_set.add(id)
            if len(image_ids_set) > previous_id_len:
                image_paths.append(path)
                image_ids.append(id)
                previous_id_len = len(image_ids)
        else:
            image_paths.append(path)
            image_ids.append(id)
        captions.append(caption)


    # group_list = list(zip(ids, captions))
    # random.shuffle(group_list)
    # ids, captions = zip(*group_list)
    print('Data is loaded.')

    return image_paths, captions, image_ids

class MSCOCODataset():
    '''
        Utilities for image captioning dataset of MSCOCO

        Initialize:
        # dt = MSCOCODataset()

        Download MSCOCO Dataset:
        # dt.download_dataset()

        Load captions and ids:
        # tid, tc, vid, vc = dt.load_data()
    '''
    def __init__(self) -> None:

        # name of the dataset.
        self.name = "MSCOCO"

        # url links for the dataset train images in zip format.
        self.train_path = "http://images.cocodataset.org/zips/train2017.zip"

        # url links for the dataset val images in zip format.
        self.val_path = "http://images.cocodataset.org/zips/val2017.zip"

        # url links for the dataset annotations in zip format.
        self.annotations_path = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

        # Project root Path
        self.root_folder = Path('.')

        # dataset Path
        self.dataset_folder = self.root_folder / "MSCOCO"

        # train images Path
        self.train_folder = self.dataset_folder / "train2017"

        # val images Path
        self.val_folder = self.dataset_folder / "val2017"

        # train images Path
        self.train_features_folder = self.dataset_folder / "features_train"

        # val features Path
        self.val_features_folder = self.dataset_folder / "features_val"

        # annotations Path
        self.annotations_folder = self.dataset_folder / "annotations"

        # train captions path
        self.train_captions = self.annotations_folder / "captions_train2017.json"

        # val captions path
        self.val_captions = self.annotations_folder / "captions_val2017.json"

    def download_dataset(self) -> None:
        '''
            Download the dataset's train images, val images and their annotations into the MSCOCO folder.
        '''

        # dataset paths in a list.
        download_list = [self.train_path, self.val_path, self.annotations_path]

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
            if ann != self.train_captions and ann != self.val_captions:
                # Check if the json file exists.
                if ann.exists():
                    # Delete the json file.
                    ann.unlink()

    def load_data(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        '''
            Load the MSCOCO captions and their corresponding image ids.
            train_ids, train_captions, val_ids, val_captions
        '''

        train_paths, train_captions, train_ids = load_json_list(self.train_captions)
        val_paths, val_captions, val_ids = load_json_list(self.val_captions, Val=True)
        test_data = None
        
        train_data = zip(train_paths, train_captions, train_ids)
        val_data = zip(val_paths, val_captions, val_ids)

        return train_data, val_data, test_data
