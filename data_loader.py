import numpy as np
from tqdm import tqdm
from experimental_data import Extract_Data

class Load_data:
    def __init__(self, processed_dir):
        self.processed_dir=processed_dir
    def load_data(self):
        features = []
        tags = []
        print("Loading data ...")
        with open(self.processed_dir+"/all_positive_data", "r") as f:
            for data in tqdm(f, total=sum(1 for _ in open(self.processed_dir+"/all_positive_data", "r"))):
                s_data=data.split("\t")
                feature = []
                for i in range(len(s_data)):
                    feature.append(int(s_data[i]))
                features.append(feature)
                tags.append(1)

        with open(self.processed_dir+"/all_negative_data", "r") as f:
            for data in tqdm(f, total=sum(1 for _ in open(self.processed_dir+"/all_negative_data", "r"))):
                s_data=data.split("\t")
                feature = []
                for i in range(len(s_data)):
                    feature.append(int(s_data[i]))
                features.append(feature)
                tags.append(0)

        return features,tags


if __name__ == "__main__":
    processed_dir="./data/processed/"
    predication_dir = "./data/SemmedDB"
    TTD_dir = "./data/TTD"

    extracting_data=Extract_Data(predication_dir,TTD_dir,processed_dir)
    extracting_data.construct_all_data()

    loading_data=Load_data(processed_dir)
    train,tags=loading_data.load_data()

