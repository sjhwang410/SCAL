from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        data = self.images[idx][:]
        data = self.transform(data)
        return data, self.labels[idx]