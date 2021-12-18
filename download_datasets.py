from prepare_msvd_dataset import *
from prepare_msrvtt_dataset import *

def downloadDatasets():

    #MSVDDataset().download_dataset()
    MSRVTTDataset().download_dataset()

downloadDatasets()
