import os
import time
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from CD_dataset import *
from CD_train_test import *
from swinsunet_channel import SwinSUNet

def get_model(in_chans = 3, encoder_stage3_blocks_num = 6):
    model = SwinSUNet(img_size=PATCH_SIZE,
                                patch_size=4,
                                in_chans=in_chans,
                                num_classes=2,
                                embed_dim=96,
                                depths=[2, 2, encoder_stage3_blocks_num, 2],
                                num_heads=[3, 6, 12, 24],
                                window_size=7,
                                mlp_ratio=4,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.2,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False,
                                up_depths=[encoder_stage3_blocks_num,2,2],
                                up_num_heads=[12,6,3])

    checkpoint = torch.load('swin_tiny_patch4_window7_224.pth', map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)

    params = {}#change the tpye of 'generator' into dict
    for name,param in model.named_parameters():
        params[name] = param.detach()

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'layers' in name and 'blocks' in name:
                if 'layers.6' in name:
                    ori_tag = 'layers.6'
                    tag = 'layers.0'
                elif 'layers.5' in name:
                    ori_tag = 'layers.5'
                    tag = 'layers.1'
                elif 'layers.4' in name:
                    ori_tag = 'layers.4'
                    tag = 'layers.2'
                else:
                    continue
                
                copy_name = name.replace(ori_tag, tag)
                parameter.copy_(params[copy_name])
    model.cuda()
    return model

def load_data(need_aug_data, train_path, val_path, patch_size, train_stride, batch_size):
    if need_aug_data:
        data_transform = tr.Compose([RandomFlip(), RandomRot()])
    else:
        data_transform = None
    train_dataset = None
    train_loader = None
    val_dataset = None
    weights = torch.from_numpy(np.ones(2))
    
    train_dataset = ChangeDetectionDataset(train_path, patch_size = patch_size, stride = train_stride, transform=data_transform)
    weights = torch.FloatTensor(train_dataset.weights).cuda()
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)
    val_dataset = ChangeDetectionDataset(val_path, patch_size = patch_size, stride = train_stride)

    return train_dataset, train_loader,  val_dataset, weights

def get_save_path(save_path):
    save_path0 = save_path
    i = 1
    while os.path.exists(save_path0):
        save_path0 = save_path + str(i)
        i=i+1
    save_path = save_path0
    save_path = save_path + '/'
    os.makedirs(save_path)
    return save_path

def get_parsms():
    argparser = argparse.ArgumentParser(description='training')
    argparser.add_argument("--path", "-p", help="path of training dataset")
    argparser.add_argument("--num_epochs", "-ne", default=200, help="number of epochs", type = int)
    argparser.add_argument("--batch_size", "-bs", default=24, help="batch_size", type = int)
    argparser.add_argument("--LR", "-lr", default=0.00001, help="lr", type = float)
    argparser.add_argument("--save_path", "-sp", default='swin_result', help="save path")
    argparser.add_argument("--gpu_id", "-gid", default=0, help="gpu id", type = int)

    args = argparser.parse_args()
    return args

if __name__ == "__main__":

    args = get_parsms()

    train_path = args.path + 'train/'
    val_path = args.path + 'val/'
    print(train_path, val_path)
    net_name = 'swin'
    IN_CHANNEL = 6
    encoder_stage3_blocks_num = 6
    save_path = get_save_path(args.save_path)

    torch.cuda.set_device(args.gpu_id)
    model = get_model(IN_CHANNEL, encoder_stage3_blocks_num)
    torch.cuda.empty_cache()

    train_dataset, train_loader, val_dataset, weights = load_data(True, \
        train_path, val_path, PATCH_SIZE, PATCH_SIZE, args.batch_size)
    criterion = nn.NLLLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=1)

    out_dic, best_f1_net_name = train(optimizer, scheduler, save_path, model, train_loader, \
        criterion, train_dataset, val_dataset, net_name, args.num_epochs)
