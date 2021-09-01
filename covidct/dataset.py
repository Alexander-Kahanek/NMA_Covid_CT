import torch, os
from torch.utils.data import Dataset, DataLoader
from glob import glob
from PIL.Image import open as open_image
from random import shuffle
from torchvision.transforms import (
    Compose,
    Resize,
    Normalize,
    ToTensor,
)


def compute_stats(dataset: Dataset):
    """Function to compute the mean and standard deviation across image channels.

    Assumes RGB images (3 channel).

    Args:
        dataset (torch.utils.data.Dataset)

    Returns:
        mean (torch.tensor): 3 element tensor containing dataset mean for each RGB channel
        std (torch.tensor): 3 element tensor containing dataset std for each RGB channel
    """

    total_sum = torch.zeros(3)
    total_sum_sq = torch.zeros(3)
    dataloader = DataLoader(dataset, batch_size=256)
    for data in dataloader:
        imgs, _ = data
        total_sum += imgs.sum(axis=[0, 2, 3])
        total_sum_sq += (imgs ** 2).sum(axis=[0, 2, 3])
    count = len(dataset) * imgs.size(2) * imgs.size(3)
    mean = total_sum / count
    std = torch.sqrt(total_sum_sq / count - mean ** 2)
    return mean, std


class CovidCTDataset(Dataset):
    """Custom PyTorch Dataset for https://www.kaggle.com/mloey1/covid19-chest-ct-image-augmentation-gan-dataset.
    
    Attributes:
        data_dir (str): path to directory containing image files.
        data (list of tuple): tuples containing img and label
        transforms (list of torchvision.transforms.Compose objects): The first transform is mandatory
            while the second may or may not be used.
    """

    def __init__(
        self,
        base_path: str,
        split: str,
        with_aug: bool = False,
        with_cgan: bool = False,
        required_transform: Compose = Compose(
            [
                ToTensor(),
                Resize((256, 256)),
                Normalize((0.5972, 0.5969, 0.5966), (0.3207, 0.3207, 0.3207)),
            ]
        ),
        optional_transform: Compose = None,
    ):
        """
        Args:
            base_path (str): path to directory containing the four data augmentation schemes.
            split (str): "train", "val", or "test".
            with_aug (bool): if True include dataset authors original augmentations.
            with_cgan (bool): if True include dataset author generative model (CGAN) augmentations.
            transform (torchvision.transform.Compose object): Mandatory transformations (e.g. ToTensor,
                Resize, and Normalize).
            transform2 (torchvision.transform.Compose object): Optional additional transformation including
                but not limited to data augmentation.
        """

        assert split in [
            "train",
            "val",
            "test",
        ], "Please provide a supported split from 'train', 'val', 'test'"
        self.split = split

        self.data_dir = os.path.join(base_path, "COVID-19")
        if with_aug:
            self.data_dir += "Aug"
        if with_cgan:
            self.data_dir += "CGAN"

        subdir = os.listdir(self.data_dir)[0]
        self.data_dir = os.path.join(self.data_dir, subdir)

        splits = os.listdir(self.data_dir)
        for s in splits:
            if self.split in s:
                self.data_dir = os.path.join(self.data_dir, s)

        covid_img_dir = os.path.join(self.data_dir, "COVID")
        noncovid_img_dir = os.path.join(self.data_dir, "NonCOVID")

        covid_imgs = glob(os.path.join(covid_img_dir, "*"))
        noncovid_imgs = glob(os.path.join(noncovid_img_dir, "*"))
        all_imgs = covid_imgs + noncovid_imgs
        labels = torch.cat(
            [
                torch.ones(len(covid_imgs), dtype=torch.long),
                torch.zeros(len(noncovid_imgs), dtype=torch.long),
            ]
        )
        self.data = list(zip(all_imgs, labels))
        shuffle(self.data)

        self.required_transform = required_transform
        self.optional_transform = optional_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        img, label = self.data[idx]
        img = open_image(img).convert(mode="RGB")
        if self.required_transform:
            img = self.required_transform(img)
        if self.optional_transform:
            img = self.optional_transform(img)
        return img, label

