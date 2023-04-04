from torch.utils.data import Dataset
import pandas as pd

class EOTDataset(Dataset):
    def __init__(self, labels_path, audio_path):
        self.labels = pd.read_xml(labels_path)
        # TODO: preprocess data
        # TODO: get a CSV that has the audio file name as the first column, and the corresponding probabilities as the remaining -> use it as labels
        # TODO: get a CSV that has the audio file name as the first column, and the input features as the remaining -> use it as data



    def __len__(self):
        # TODO: return the lenght of the X array
        return len([])

    def __getitem__(self, idx):
        # TODO: return both X and Y vectors
        # TODO: use the
        return 'hoi'

