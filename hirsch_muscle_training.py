import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, utils
from einops import rearrange
import os
from torchvision.transforms import RandomRotation
import math
import csv
from histo_vit import vit_small
import random
from torchvision.transforms.functional import hflip
from torchvision.transforms.functional import vflip
import segmenter
import og_mae


def augment_image_with_map(_img, _map):
    x_start = torch.randint(low=0, high=(256-224), size=(1,)).item()
    y_start = torch.randint(low=0, high=(256-224), size=(1,)).item()
    _img = _img[:, :, x_start:(x_start + 224), y_start:(y_start + 224)]
    _map = _map[:, x_start:(x_start + 224), y_start:(y_start + 224)]

    if torch.rand(1).item() < 0.5:
        _img = hflip(_img)  # horizontal flip
        _map = hflip(_map)  # horizontal flip

    if torch.rand(1).item() < 0.5:
        _img = vflip(_img)  # vertical flip
        _map = vflip(_map)  # vertical flip

    random_rotation = random.choice([RandomRotation((0, 0)),
                                     RandomRotation((90, 90)),
                                     RandomRotation((-90, -90)),
                                     RandomRotation((180, 180))])

    _img = random_rotation(_img)  # rotate
    _map = random_rotation(_map)  # rotate
    return _img, _map


def adjust_learning_rate(epoch, sched_config):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < sched_config['warmup_epochs']:
        lr = sched_config['lr'] * epoch / sched_config['warmup_epochs']
    else:
        lr = sched_config['min_lr'] + (sched_config['lr'] - sched_config['min_lr']) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - sched_config['warmup_epochs']) / (sched_config['epochs'] - sched_config['warmup_epochs'])))
    return lr

def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

file_names = ['S14-580.pt',
              'S00-1910.pt',
              'S02-410.pt',
              'S02-484.pt',
              'S03-2391.pt',
              'S01-18.pt',
              "S03-3178 D2.pt",
              "S03-3178 D3.pt",
              "S03-3178 D4.pt",
              'S04-52.pt',
              'S04-910.pt',
              'S07-1808.pt',
              'S08-2215.pt',
              'S09-2723.pt',
              'S04-1840.pt',
              'S07-1465.pt',
              'S14-1715.pt',
              'S09-2909.pt',
              'S14-3414.pt',
              'S14-2038.pt',
              'S15-1442.pt',
              'S15-1518.pt',
              'S16-567.pt',
              "S16-1197 B1.pt",
              'S11-1760.pt',
              'S16-1467.pt',
              "S16-1197 B3.pt",
              "S16-1197 B2.pt",
              'S97-2054.pt',
              'S16-1415.pt']

for fold_num in range(5):

    validation_files = file_names[(6*fold_num):(6*fold_num+6)]

    train_imgs = []
    train_labels = []
    val_imgs = []
    val_labels = []

    data_paths = os.listdir('/windows/histo/muscle_5x_normed')
    for i_path, data_path in enumerate(data_paths):
        torch_obj = torch.load(f'/windows/histo/muscle_5x_normed/{data_path}')

        if data_path in validation_files:
            val_imgs.append(torch_obj['imgs'])
            val_labels.append(torch_obj['muscles'])
        else:
            train_imgs.append(torch_obj['imgs'])
            train_labels.append(torch_obj['muscles'])

    train_imgs = torch.cat(train_imgs, dim=0)  # (48_000, 3, 256, 256)
    train_labels = torch.cat(train_labels, dim=0)  # (48_000, 1, 256, 256)
    val_imgs = torch.cat(val_imgs, dim=0)  # (12_000, 3, 256, 256)
    val_labels = torch.cat(val_labels, dim=0)  # (12_000, 1, 256, 256)

    print(f'Starting fold: {fold_num}')
    print(train_imgs.shape, train_labels.shape)
    print(val_imgs.shape, val_labels.shape)

    batch_size = 96
    train_loader = DataLoader(TensorDataset(train_imgs, train_labels), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_imgs, val_labels), batch_size=batch_size, shuffle=False)

    del train_imgs
    del train_labels
    del val_imgs
    del val_labels

    best_val_loss = 10
    base_lr = 1e-4
    learning_rate = base_lr * batch_size / 256

    model = og_mae.mae_vit_base_patch16_dec512d8b().cuda()
    model.load_state_dict(torch.load('mae_visualize_vit_base.pth')['model'])
    linear = nn.Linear(768, 512).cuda()

    # decoder = segmenter.MaskTransformer(n_cls=2,
    #                                     patch_size=16,
    #                                     d_encoder=384,
    #                                     n_layers=2,
    #                                     n_heads=12,
    #                                     d_model=384,
    #                                     d_ff=1536,
    #                                     drop_path_rate=0,
    #                                     dropout=0)
    # seg_head = segmenter.Segmenter(decoder=decoder, n_cls=2).cuda()

    # optimizer
    backbone_params = model.parameters()
    linear_params = linear.parameters()
    # head_params = seg_head.parameters()
    opt = torch.optim.AdamW([{'params': backbone_params}, {'params': linear_params}], lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()

    # Prep LR stepping
    epochs = 50
    multiplier = 1
    backbone_config = {'lr': learning_rate,
                       'warmup_epochs': 5,
                       'min_lr': 0,
                       'epochs': epochs}

    head_config = {'lr': multiplier * learning_rate,
                   'warmup_epochs': 5,
                   'min_lr': 0,
                   'epochs': epochs}
    num_down = 0
    for epoch in range(epochs):
        if num_down >= 20:
            break

        opt.param_groups[0]['lr'] = adjust_learning_rate(epoch, backbone_config)
        opt.param_groups[1]['lr'] = adjust_learning_rate(epoch, head_config)

        current_lr_backbone = opt.param_groups[0]['lr']  # confirm
        current_lr_head = opt.param_groups[1]['lr']  # confirm

        train_losses = []

        model = model.train()
        # seg_head = seg_head.train()
        linear = linear.train()
        for batch in train_loader:
            img, plexus = batch  # load from batch
            img, plexus = augment_image_with_map(img.cuda(), plexus.cuda())  # perform data augmentation

            img = img.to(dtype=torch.bfloat16) / 255  # (bsz, 3, H, W)
            plexus = plexus.long()  # (bsz, H, W)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                x = model.patch_embed(img)
                x = x + model.pos_embed[:, 1:, :]

                cls_token = model.cls_token + model.pos_embed[:, :1, :]
                cls_tokens = cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)

                # apply Transformer blocks
                for blk in model.blocks:
                    x = blk(x)  # (bsz, L, 768)

                x = linear(x)  # (bsz, L, 512)
                logits = rearrange(x[:, 1:, :], 'b (h w) (c i j) -> b c (h i) (w j)', h=14, w=14, c=2, i=16, j=16)  # (bsz, 2, H, W)
                # logits = seg_head(features=x[:, 1:, :], HW_input=224, HW_target=224)  # (bsz, 2, H, W)


            loss = loss_function(logits, plexus)
            loss.backward()
            opt.step()
            opt.zero_grad()
            train_losses.append(loss.item())

        val_losses = []
        model.eval()
        for batch in val_loader:
            img, plexus = batch  # load from batch
            img = img.cuda().to(dtype=torch.bfloat16) / 255  # (bsz, 3, H, W)
            plexus = plexus.cuda().long()  # (bsz, H, W)

            x_start = torch.randint(low=0, high=(256 - 224), size=(1,)).item()
            y_start = torch.randint(low=0, high=(256 - 224), size=(1,)).item()
            img = img[:, :, x_start:(x_start + 224), y_start:(y_start + 224)]
            plexus = plexus[:, x_start:(x_start + 224), y_start:(y_start + 224)]

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    x = model.patch_embed(img)
                    x = x + model.pos_embed[:, 1:, :]

                    cls_token = model.cls_token + model.pos_embed[:, :1, :]
                    cls_tokens = cls_token.expand(x.shape[0], -1, -1)
                    x = torch.cat((cls_tokens, x), dim=1)

                    # apply Transformer blocks
                    for blk in model.blocks:
                        x = blk(x)  # (bsz, L, 768)

                    x = linear(x)  # (bsz, L, 512)
                    logits = rearrange(x[:, 1:, :], 'b (h w) (c i j) -> b c (h i) (w j)', h=14, w=14, c=2, i=16,
                                       j=16)  # (bsz, 2, H, W)
                    # logits = seg_head(features=x[:, 1:, :], HW_input=224, HW_target=224)  # (bsz, 2, H, W)

            loss = loss_function(logits, plexus)
            val_losses.append(loss.item())

        train_losses = torch.Tensor(train_losses).mean().item()
        val_losses = torch.Tensor(val_losses).mean().item()
        print(f'Epoch: {epoch}, Train Loss: {train_losses}, Val Loss: {val_losses}, LR Backbone: {current_lr_backbone}, LR Head: {current_lr_head},')

        if best_val_loss > val_losses:
            best_val_loss = val_losses
            print(f'SAVING')
            # torch.save(obj={'backbone': model.state_dict(),
            #                 'head': seg_head.state_dict()},
            #            f=f'saved_models/ViT_HIPT_{fold_num}_muscle_5x_{base_lr}.pt')
            torch.save(obj={'backbone': model.state_dict(),
                            'linear': linear.state_dict()},
                       f=f'saved_models/ViT_IN1k_{fold_num}_muscle_5x_{base_lr}.pt')
            num_down = 0
        else:
            num_down += 1

        # write to logs
        with open(f'ViT_IN1k_muscle_logs_5x_{base_lr}.csv', 'a', errors="ignore") as out_file:
            csv_writer = csv.writer(out_file, delimiter=',', lineterminator='\n')
            csv_writer.writerow([epoch, train_losses, val_losses, best_val_loss, current_lr_backbone, current_lr_head, base_lr, fold_num])







