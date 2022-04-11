from torch.utils.data import Dataset
import pandas as pd
import torch


class ApolloDataset(Dataset):
    def __init__(self, is_train):
        self.TRAIN_RATIO = 2
        self.is_train = is_train
        self.file_location = "kidney_disease.csv"
        csv_data = pd.read_csv("kidney_disease.csv")
        self.total = len(csv_data)

        df = pd.DataFrame(csv_data)

        self.test_count = len(df) // self.TRAIN_RATIO
        if len(df) % self.TRAIN_RATIO != 0:
            self.test_count += 1
        self.train_count = len(df) - self.test_count
        self.count = self.train_count
        if self.is_train is False:
            self.count = self.test_count

        self.samples = torch.zeros((self.count, 3))
        self.targets = torch.zeros(self.count, dtype=torch.long)

        current_row_index = 0
        for index, row in df.iterrows():
            mod = index % self.TRAIN_RATIO
            if is_train and mod == 0:
                continue
            if is_train is False and mod != 0:
                continue
            self.targets[current_row_index] = row[len(row)-1]
            for i in range(len(row)-1):
                self.samples[current_row_index, i] = row[i]
            current_row_index = current_row_index + 1

    def _length(self, var):
        if torch.is_tensor(var):
            return var.shape[0]
        else:
            return 1

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]


if __name__ == "__main__":
    d = ApolloDataset(is_train=True)
    from torch.utils.data import DataLoader
    dl = DataLoader(d, batch_size=3)
    for x,y in dl:
        print(x.shape)


