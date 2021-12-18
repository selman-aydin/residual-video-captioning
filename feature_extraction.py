# Tries for dataloader and dataset on feature extraction.

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from torchvision.io import read_video
from Inception import inception_v3
from typing import List, Union

from pathlib import Path
from prepare_msrvtt_dataset import MSRVTTDataset
from prepare_msvd_dataset import MSVDDataset
from tqdm import tqdm

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
PARAMS = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}

model = inception_v3(True)
model = model.to(device)

IMAGE_SIZE = 299
FRAME_SIZE = 8

msrvtt_dt = MSRVTTDataset()
msvd_dt   = MSVDDataset()

class FeatureExtractionDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    '''
        A Torch Dataset utilized to Load Video's visual and audial data.
        inputs are:
        ids: list of paths where the video files are.
        if the ids are consist of only names.
        path: to determine where are the files.
        im_size: To transform the image into specified size. ie: inception_v3 model takes 3x299x299 so the im_size = 299.
        frame_size: how many visual frame to be extracted from the visual data.
    '''
    def __init__(self, ids, path, im_size, frame_size):
        'Initialization'
        self.ids = ids
        self.path = path
        self.transform = Compose([
            Resize(im_size),
            CenterCrop(im_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.im_size = im_size
        self.frame_size = frame_size
        

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.ids)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        path = self.ids[index]
        # Load data as visual, audial, and get fps of both visual and audial.
        visual, audial, info = read_video(str(path))

        # initialize visual frames with zeros.
        visual_frames = torch.zeros((self.frame_size, 3, self.im_size, self.im_size))
        # take how many frames are there in the visual data.
        frame_length = visual.shape[0]
        # determine what to be the step size (for equal sized splitting) so that the frame size for every visual data to be the same.
        step_size = frame_length // self.frame_size
        # for loop to get every determined frame with determined step size.
        for i in range(self.frame_size):
            # get which frame to be taken.
            idx = i * step_size
            # get the frame from the visual data.
            im = visual[idx]
            # Change order of the channels and normalize to 0-1. shape (H, W, C) -> (C, H, W)
            im = torch.div(im, 255).permute(2, 0, 1) 
            # Transform the images into given im size and normalize.
            im = self.transform(im)
            # add the resulted frame into the corresponding visual frames index.
            visual_frames[i] = im
            
        id = str(path.parts[-1][:-4]) # get ID of the image from its path.

        return visual_frames, id

class FeatureExtraction():
    '''
    Class for visual and audial feature extraction.
    '''
    def __init__(self, input_folder:Path, output_folders:List[Union[Path, Path]], model:torch.nn.Module, IMAGE_SIZE:int, FRAME_SIZE:int, params:dict) -> None:

        self.input_folder = input_folder # input video folder
        self.output_visual_folder = output_folders[0] # output folder for visual features
        self.model = model.eval() # assign model in evaluation mode
        self.IMAGE_SIZE = IMAGE_SIZE  # initialize input image size
        self.params = params # initialize parameters for dataloader
        dt_name = str(self.output_visual_folder).split("/")[-2]
        if dt_name == "MSVD":
            self.ids = list(input_folder.glob('*.avi'))  # list of video files from input folder
        else:
            self.ids = list(input_folder.glob('*.mp4'))  # list of video files from input folder
        self.frame_size = FRAME_SIZE # initialize frame size
        self.set = FeatureExtractionDataset(self.ids, self.input_folder, self.IMAGE_SIZE, self.frame_size) # initialize dataset
        self.generator = DataLoader(self.set, **self.params) # initialize dataloader
        # create output folders
        if not self.output_visual_folder.exists():
            Path(self.output_visual_folder).mkdir(parents=True, exist_ok=True)
            print(f"path: {str(self.output_visual_folder)} is created.")


    def run(self) -> None:
        '''
            Scan through generator, extract features from visual data get audial features
            from videos and save them in .pt files in corresponding output folders.
        '''
        print(f"Extracting: {str(self.input_folder)}")
        for local_visual, local_ids in tqdm(self.generator):
            local_visual = local_visual.to(device)
            visual_features = torch.zeros((local_visual.shape[0], self.frame_size, 2048))
            for i in range(self.frame_size):
                with torch.no_grad():
                    inp = local_visual[:, i, :, :, :]
                    visual_features[:, i, :] = model(inp) # Shape (N, frame_size, feature_size)
            for visual_feature, id in zip(visual_features, local_ids):
                torch.save(visual_feature.cpu(), self.output_visual_folder / f'{id}.pt')
        print(f"{str(self.input_folder)} extracted.")

input_folders = [msvd_dt.train_folder,msvd_dt.val_folder,msvd_dt.test_folder,msrvtt_dt.train_folder, msrvtt_dt.test_folder]
output_folders = [[msvd_dt.train_features_folder],[msvd_dt.val_features_folder],[msvd_dt.test_features_folder],[msrvtt_dt.train_features_folder], [msrvtt_dt.test_features_folder]]

for inp, out in zip(input_folders, output_folders):
    x = FeatureExtraction(input_folder=inp, output_folders=out, model=model, IMAGE_SIZE=IMAGE_SIZE, FRAME_SIZE=FRAME_SIZE, params=PARAMS)
    x.run()
