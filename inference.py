import torch
from torchvision.utils import save_image
from prepare_dataset import create_data, single_image_preprocess, CamVidDataset, label_to_rgb_tensor
import random
from networks.network import UnetGenerator, UNet
import matplotlib.pyplot as plt
import os

def generate_image(dataroot: str, num: int):
    """
    Generates images randomly and saves under `generated_image` directory.

    args:
        dataroot (str): cityscapes or camvid
        num (int): Number of images to generate
    """
    # Create generated images folder if not exists
    os.makedirs('generated_images', exist_ok=True)

    # Create an instance of the generator
    generator = UnetGenerator(3, 3, 64, use_dropout=False)

    # Checks the dataroot and load the model
    if dataroot == 'cityscapes':
        checkpoint = torch.load(f'checkpoints/{dataroot}/model_cityscapes_color_epoch_50.pth')
    elif dataroot == 'camvid':
        checkpoint = torch.load(f'checkpoints/{dataroot}/model_epoch_50.pth')
    generator.load_state_dict(checkpoint['gen'])

    # Create the dataset for inference
    train, val = create_data(dataroot=dataroot)

    # Generate the images
    ls = []
    for i in range(num):
        idx = int(torch.randint(0, len(val), (1,)))
        realA, realB = val[idx]
        generator.eval()
        with torch.inference_mode():
            fakeB = generator(realA.unsqueeze(dim=0))
        
        # Saves the images
        save_image(fakeB.squeeze(), f'generated_images/gen_fakeB_{i}.png')
        save_image(realA, f'generated_images/gen_realA_{i}.png')

        # List containing the path of the images used in flask html
        ls.append({
            'fakeB': f'generated_images/gen_fakeB_{i}.png',
            'realA': f'generated_images/gen_realA_{i}.png',
        })
    return ls

def predict_image(dataroot, img_path):
    # Create generated images folder if not exists
    os.makedirs('generated_images', exist_ok=True)

    # Create an instance of the generator
    generator = UnetGenerator(3, 3, 64, use_dropout=False)

    # Checks the dataroot and load the model
    if dataroot == 'cityscapes':
        checkpoint = torch.load('checkpoints/cityscapes/model_cityscapes_color_epoch_50.pth')
    elif dataroot == 'camvid':
        checkpoint = torch.load('checkpoints/camvid/model_epoch_50.pth')
    generator.load_state_dict(checkpoint['gen'])

    realA = single_image_preprocess(img_path)
    with torch.inference_mode():
        fakeB = generator(realA.unsqueeze(dim=0))
        
    # Saves the images
    save_image(fakeB.squeeze(), f'generated_images/gen_fakeB.png')
    save_image(realA, f'generated_images/gen_realA.png')

    # List containing the path of the images used in flask html
    imgs = [{
        'fakeB': f'generated_images/gen_fakeB.png',
        'realA': f'generated_images/gen_realA.png',
    }]
    return imgs

def segement_image(dataroot, img_path):
    # Create generated images folder if not exists
    os.makedirs('generated_images', exist_ok=True)

    # Create an instance of the generator
    model = UNet(in_channels=3, num_classes=32, depth=5)

    # Checks the dataroot and load the model
    if dataroot == 'cityscapes':
        checkpoint = torch.load('checkpoints/models50.pth')
    elif dataroot == 'camvid':
        checkpoint = torch.load('checkpoints/models50.pth')
    model.load_state_dict(checkpoint['model'])

    realA = CamVidDataset(img_path)[0]
    print(realA.shape)
    with torch.no_grad():
        outputs = model(realA.unsqueeze(0))
        preds = torch.argmax(outputs, dim=1)
    print(preds.shape)
    pred_rgb = label_to_rgb_tensor(preds[0]).cpu().numpy().transpose(1, 2, 0)
    print
    # Saves the images
    plt.imsave(f'generated_images/segment.png', pred_rgb)
    save_image(realA, f'generated_images/real.png')

    # List containing the path of the images used in flask html
    imgs = [{
        'segment': f'generated_images/segment.png',
        'real': f'generated_images/real.png',
    }]
    return imgs