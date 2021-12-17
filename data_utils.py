from torch import load, LongTensor
from torch.utils.data import Dataset, DataLoader
from text_processing import get_vocab

PARAMS = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}


class TestDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, dt_name, paths, feature_folder):
        'Initialization'
        self.name = dt_name
        self.paths = paths
        self.feature_folder = feature_folder

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.paths)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        name = self.paths[index] + ".pt"
        feature = load(self.feature_folder / name, map_location='cpu')
        if self.name == "MSVD":
            id = self.paths[index]
        else:
            id = int(self.paths[index][5:])

        return feature, id


class TrainDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, ids, captions, feature_folder):
        'Initialization'
        self.ids = ids
        self.captions = captions
        self.feature_folder = feature_folder


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.ids)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        name = self.ids[index] + ".pt"
        feature = load(self.feature_folder / name, map_location='cpu')

        tokens = self.captions[index]
        tokens = LongTensor(tokens)

        return feature, tokens


def get_loader_and_vocab(dt):
    train_data, val_ids, test_ids = dt.load_data()
    processed_paths, processed_captions, vocab, tokenizer = get_vocab(train_data)
    train_dataset = TrainDataset(ids=processed_paths, captions=processed_captions,
                                 feature_folder=dt.train_features_folder)
    train_loader = DataLoader(train_dataset, **PARAMS)
    val_dataset = TestDataset(dt_name=dt.name, paths=val_ids, feature_folder=dt.val_features_folder)
    val_loader = DataLoader(val_dataset, **PARAMS)
    test_dataset = TestDataset(dt_name=dt.name, paths=test_ids, feature_folder=dt.test_features_folder)
    test_loader = DataLoader(test_dataset, **PARAMS)

    return train_loader, val_loader, test_loader, vocab