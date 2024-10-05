from torch.utils.data import  Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import torch
import torchvision.transforms.functional as transf_F

class_dict_path = 'camvid_data/class_dict.csv'
device = 'cuda'

class CustomDataTrain(Dataset):
    def __init__(self, inp, tar, transform):
        self.inp = inp
        self.tar = tar
        self.transform = transform

    def __len__(self):
        return len(self.inp)
    
    def __getitem__(self, idx):
        inp = self.inp[idx]
        tar = self.tar[idx]
        inp = Image.open(inp).convert("RGB")
        tar = Image.open(tar).convert("RGB")
        
        image_a = self.transform(inp)
        image_b = self.transform(tar)
        return image_a, image_b


def create_data(dataroot: str):
    """
    Creates the `torch.utils.data.Datasets` for the appropriate dataroot.
    arg:
        dataroot (str): cityscapes or camvid
    """

    # Initialize the paths of train and val data
    if dataroot == 'cityscapes':
        X_TRAIN_DIR = 'cityscapes_data/train/'
        Y_TRAIN_DIR = 'cityscapes_data/train_labels/'
        X_VAL_DIR = 'cityscapes_data/val/'
        Y_VAL_DIR = 'cityscapes_data/val_labels/'
    elif dataroot == 'camvid':
        X_TRAIN_DIR = 'camvid_data/train/'
        Y_TRAIN_DIR = 'camvid_data/train_labels/'
        X_VAL_DIR = 'camvid_data/val/'
        Y_VAL_DIR = 'camvid_data/val_labels/'

    # Get all the image paths
    X_train = [X_TRAIN_DIR+x for x in os.listdir(X_TRAIN_DIR) if x.endswith('.png')]
    y_train = [Y_TRAIN_DIR+x for x in os.listdir(Y_TRAIN_DIR) if x.endswith('.png')]
    X_val = [X_VAL_DIR+x for x in os.listdir(X_VAL_DIR) if x.endswith('.png')]
    y_val = [Y_VAL_DIR+x for x in os.listdir(Y_VAL_DIR) if x.endswith('.png')]

    # Define tranforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # val_transform = transforms.Compose([
    #     transforms.Resize((256, 256)),
    #     transforms.ToTensor(),
    # ])

    # Create the torch Datasets
    train_dataset = CustomDataTrain(inp=y_train, tar=X_train, transform=train_transform)
    val_dataset = CustomDataTrain(inp=y_val, tar=X_val, transform=train_transform)

    return train_dataset, val_dataset

def single_image_preprocess(img_path):
    """
    Applies resize(256, 256) and convert an image into tensor.
    arg:
        img (str): Image path
    """
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    return transform(img)

# Load class dictionary
class_dict: pd.DataFrame = pd.read_csv(class_dict_path)

# Create conversion dictionaries
rgb_to_label_dict: dict[tuple, int] = {
    (row['r'], row['g'], row['b']): idx
    for idx, row in class_dict.iterrows()
}

label_to_rgb_dict: dict[int, tuple] = {
    idx: (row['r'], row['g'], row['b'])
    for idx, row in class_dict.iterrows()
}

def rgb_to_label(image: torch.Tensor) -> torch.Tensor:
    width, height, _ = image.shape
    label_image = torch.zeros(width, height, device=device)

    # Transformar para inteiro, se necessário, para garantir comparação correta
    image = (image * 255).int()

    for rgb, label in rgb_to_label_dict.items():
        rgb_tensor = torch.tensor(rgb, device=device)
        mask = torch.all(image == rgb_tensor, dim=-1)
        label_image[mask] = label
        
    return label_image

def label_to_rgb_tensor(label_tensor: torch.Tensor) -> torch.Tensor:
    height, width = label_tensor.shape
    rgb_image = torch.zeros(3, height, width, dtype=torch.uint8)

    for label, rgb in label_to_rgb_dict.items():
        mask = (label_tensor == label)
        rgb_image[0][mask] = rgb[0]  # Red
        rgb_image[1][mask] = rgb[1]  # Green
        rgb_image[2][mask] = rgb[2]  # Blue

    return rgb_image

def read_image(image_path: str) -> torch.Tensor:
    image = Image.open(image_path)
    image = transforms.ToTensor()(image)
    return image

class CamVidDataset(Dataset):
    def __init__(self, img_dir: str, augment: bool = False):
        self.img_dir = img_dir
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.Resize((360, 480)),
        ])
        self.img_files = [img_dir]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):

        img = read_image(self.img_files[idx])

        img = self.transform(img)
        
        if self.augment:
            # Apply the same random transformations to both image and label
            if torch.rand(1) > 0.5:
                img = transf_F.hflip(img)
            
            # Pad the image and label with the same padding value (using a tuple)
            img = transf_F.pad(img, (10, 10, 10, 10))  # padding on all sides
        

            # Apply the same cropping to both image and label
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(360, 480))
            img = transf_F.crop(img, i, j, h, w)
            
            # Color jitter only applies to the image
            img = transforms.ColorJitter(brightness=0.1, contrast=0, saturation=0, hue=0.2)(img)

        return img
    