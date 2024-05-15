import slideio
import os
import rasterio
import torch
import numpy as np
from torchvision.transforms import (
    Compose,
    RandomApply,
    RandomHorizontalFlip,
    RandomRotation,
    RandomVerticalFlip,
    Resize,
    InterpolationMode
)
from einops import rearrange
import torchstain

def trim_seg_mask(seg_mask):
    h, w, c = seg_mask.shape
    # left limit
    for i in range(w):
        if np.mean(seg_mask[:, i, :]) < 255:
            break

    # right limit
    for j in range(w - 1, 0, -1):
        if np.mean(seg_mask[:, j, :]) < 255:
            break

    # top limit
    for k in range(h):
        if np.mean(seg_mask[k, :, :]) < 255:
            break

    # bottom limit
    for v in range(h - 1, 0, -1):
        if np.mean(seg_mask[v, :, :]) < 255:
            break

    cropped = seg_mask[k:(v + 1), i:(j + 1), :].copy()
    return cropped


def binary_seg_mask(seg_mask):
    h,w,c = seg_mask.shape
    binary_mask = np.zeros_like(seg_mask)[:, :, 0]
    for i in range(h):
        for j in range(w):
            if np.mean(seg_mask[i, j, :]) > 250:
                binary_mask[i, j] = 1

    return binary_mask


image_paths = os.listdir('raw_images')
for i_path, image_path in enumerate(image_paths):
    # load raw WSI
    full_image_path = f'raw_images/{image_path}'
    slide = slideio.open_slide(full_image_path, 'SVS')
    scene = slide.get_scene(0)
    ds_image = torch.ByteTensor(scene.read_block())

    # load annotations
    file_id = image_path.replace('.svs', '')
    muscle_path = f'{file_id}_GTmuscMask.tif'
    plexus_path = f'{file_id}_plexinc.tif'

    with rasterio.open(muscle_path) as f:
        musc_array = f.read().transpose(1, 2, 0)

    with rasterio.open(plexus_path) as f:
        plexus_array = f.read().transpose(1, 2, 0)

    # annotations are cropped weird for some reason, we fix it here
    cropped_musc = trim_seg_mask(musc_array)
    cropped_plexus = trim_seg_mask(plexus_array)
    binary_musc = torch.ByteTensor(binary_seg_mask(cropped_musc))
    binary_plexus = torch.ByteTensor(binary_seg_mask(cropped_plexus))

    # resize to half the height and width since Macenko normalization couldn't fit into memory with full resolution
    img = rearrange(ds_image, 'h w c -> c h w')
    img = Resize(size=(int(img.shape[1]/2), int(img.shape[2]/2)), interpolation=InterpolationMode.NEAREST)(img)

    if i_path == 0:
        # Using the first WSI as the reference to train the normalizer, this can be changed
        torch_normalizer = torchstain.normalizers.MacenkoNormalizer(backend='numpy')
        torch_normalizer.fit(img.numpy())
        print(f'Normalized file: {image_path}')

    # applying the normalizer to the WSI
    img, _, _ = torch_normalizer.normalize(I=img.numpy(), stains=False)
    img = torch.ByteTensor(img)

    # resizing the annotation masks to be the same size as the WSI (so the pixels line-up)
    muscle = Resize(size=(img.shape[1], img.shape[2]), interpolation=InterpolationMode.NEAREST)(rearrange(binary_musc, 'h w -> 1 h w'))
    plexus = Resize(size=(img.shape[1], img.shape[2]), interpolation=InterpolationMode.NEAREST)(rearrange(binary_plexus, 'h w -> 1 h w'))

    print(img.shape, muscle.shape, plexus.shape)  # double check all are equal

    # save normalized WSI and annotations
    save_path = f'10x_images_segmentations/{file_id}.pt'
    torch.save(obj={'img': img,
                    'muscle': muscle,
                    'plexus': plexus},
               f=save_path)
