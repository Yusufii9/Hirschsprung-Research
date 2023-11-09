import torch
import os
from torchvision.transforms import (
    Compose,
    Resize,
    RandomCrop
)

transform_2x = Compose(
    [
        RandomCrop(size=(512, 512)),
        Resize((256, 256)),
    ]
)

data_paths = os.listdir('10x_images_segmentations')
for i_path, data_path in enumerate(data_paths):
    torch_obj = torch.load(f'10x_images_segmentations/{data_path}')
    img = torch_obj['img']  # (3, H, W)
    muscle = torch_obj['muscle']  # (1, H, W)

    imgs = []
    muscles = []

    for _ in range(2_000):
        # 2x transform
        state = torch.get_rng_state()
        img_2x = transform_2x(img)  # (3, H, W)
        torch.set_rng_state(state)
        muscle_2x = transform_2x(muscle)  # (1, H, W)

        imgs.append(img_2x.unsqueeze(dim=0))
        muscles.append(muscle_2x)

    imgs = torch.cat(imgs, dim=0)
    muscles  = torch.cat(muscles, dim=0)
    print(imgs.shape, muscles.shape)

    torch.save(obj={'imgs': imgs,
                    'muscles': muscles},
               f=f'muscle_5x_normed/{data_path}')




