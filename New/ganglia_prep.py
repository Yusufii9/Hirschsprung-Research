# import scipy
# import sklearn
# from sklearn.feature_extraction import image
from scipy.io import loadmat
import matplotlib.pyplot as plt
from einops import rearrange
import numpy as np
import mat73
import rasterio
import os
import slideio
import torch
from torchvision.transforms import (
    Compose,
    RandomApply,
    RandomHorizontalFlip,
    RandomRotation,
    RandomVerticalFlip,
    Resize,
    InterpolationMode
)

# image_paths = os.listdir('/windows/histo/raw_images')
# for i_path, image_path in enumerate(image_paths):
#     full_image_path = f'/windows/histo/raw_images/{image_path}'
#     slide = slideio.open_slide(full_image_path, 'SVS')
#     scene = slide.get_scene(0)
#     ds_image = torch.ByteTensor(scene.read_block())
#
#     file_id = image_path.replace('.svs', '')
#     ganglia = torch.ByteTensor(mat73.loadmat(f'/windows/histo/ganglia/{file_id}_certainGanglia_P1.mat')['mask'])
#     print(ganglia.shape, ds_image.shape)

    # new_h = int(plexus.shape[0] / 2)
    # new_w = int(plexus.shape[1] / 2)
    # print(new_h, new_w)
    #
    # plexus = rearrange(plexus, 'h w -> 1 h w')
    # plexus = Resize(size=(new_h, new_w), interpolation=InterpolationMode.NEAREST)(plexus)[0]
    # torch.save(obj=plexus, f=f'/windows/histo/10x_ganglia_certain/{file_id}.pt')

    # print(plexus.shape)
    #
    # ds_image = rearrange(ds_image, 'h w c -> c h w')
    # ds_image = Resize(size=(new_h, new_w), interpolation=InterpolationMode.NEAREST)(ds_image)
    # ds_image = rearrange(ds_image, 'c h w -> h w c')
    #
    # print(ds_image.shape)
    #
    # plt.rcParams['figure.figsize'] = [20, 20]
    # plt.imshow(ds_image, cmap='gray')
    # plt.imshow(plexus[0], cmap='jet', alpha=0.25)
    #
    # plt.savefig(f'/home/hammy/histo_predictions/mat_for_{file_id}.png')  # save the figure to file
    # plt.close()  # close the figure window


data_paths = os.listdir('/windows/histo/10x_images_segmentations')
for i_path, data_path in enumerate(data_paths):
    torch_obj = torch.load(f'/windows/histo/10x_images_segmentations/{data_path}')
    img = torch_obj['img']  # (3, H, W)
    muscle = torch_obj['muscle']  # (1, H, W)
    plexus = torch.load(f'/windows/histo/10x_plexus_only/{data_path}').unsqueeze(dim=0)  # (1, H, W)
    ganglia_potential = torch.load(f=f'/windows/histo/10x_ganglia_potential/{data_path}').unsqueeze(dim=0)  # (1, H, W)
    ganglia_certain = torch.load(f=f'/windows/histo/10x_ganglia_certain/{data_path}').unsqueeze(dim=0)  # (1, H, W)
    ganglia = ganglia_potential + ganglia_certain
    print(ganglia.unique())
    ganglia[ganglia==2] = 1
    print(ganglia.unique())

    hw = 224
    imgs = []
    gang = []
    count = 0
    for _ in range(500_000):
        h_start = torch.randint(low=0, high=(img.shape[1] - hw), size=(1,)).item()
        h_end = h_start + hw
        w_start = torch.randint(low=0, high=(img.shape[2] - hw), size=(1,)).item()
        w_end = w_start + hw

        img_cropped = img[:, h_start:h_end, w_start:w_end]
        muscle_cropped = muscle[:, h_start:h_end, w_start:w_end]
        plexus_cropped = plexus[:, h_start:h_end, w_start:w_end]
        ganglia_cropped = ganglia[:, h_start:h_end, w_start:w_end]

        if torch.any(plexus_cropped) == 1:
            imgs.append(img_cropped.unsqueeze(dim=0))
            gang.append(ganglia_cropped)

            count += 1
            if count >= 1_000:
                break

    imgs = torch.cat(imgs, dim=0)
    gang  = torch.cat(gang, dim=0)
    print(imgs.shape, gang.shape)

    torch.save(obj={'imgs': imgs,
                    'plexes': gang},
               f=f'/windows/histo/10x_ganglia_union_w_images/{data_path}')




