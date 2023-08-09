import os
import time
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from CD_dataset import *
from CD_train_test import *
from swinsunet_channel import SwinSUNet

def get_model(trained_model, in_chans = 3, encoder_stage3_blocks_num = 6):
    model = SwinSUNet(img_size=224,
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
    model.load_state_dict(torch.load(trained_model, map_location='cpu'))
    model.cuda()
    return model

def load_data(test_path, patch_size, test_batch_size = 8):
    test_dataset = TestDataset(test_path, patch_size = patch_size)
    test_loader = DataLoader(test_dataset, batch_size = test_batch_size, shuffle = False, num_workers = 4)
    return test_dataset, test_loader

def get_parsms():
    argparser = argparse.ArgumentParser(description='prediction')
    argparser.add_argument("--path", "-p", help="path of prediction dataset")
    argparser.add_argument("--save_path", "-sp", help="path of save dataset")
    argparser.add_argument("--model_name", "-model_name", help="path of model")
    argparser.add_argument("--batch_size", "-bs", default=1, help="batch_size", type = int)
    argparser.add_argument("--gpu_id", "-gid", default=0, help="gpu id", type = int)
    args = argparser.parse_args()
    return args

if __name__ == "__main__":
    args = get_parsms()
    test_path = args.path
    IN_CHANNEL = 6
    encoder_stage3_blocks_num = 6
    torch.cuda.set_device(args.gpu_id)
    model = get_model(args.model_name, IN_CHANNEL, encoder_stage3_blocks_num)
    test_dataset, test_loader = load_data(test_path, 224, args.batch_size)

    predict_one_pic(test_dataset, model, args.save_path)