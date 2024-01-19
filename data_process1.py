import os
import argparse
import pickle
import cv2
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from PIL import Image
from ruamel.yaml import YAML
from torchvision.transforms import Grayscale, Normalize, ToTensor, transforms
from utils.helpers import dir_exists, remove_files


def data_process(data_path, name, patch_size, stride, mode):
    save_path = os.path.join(data_path, f"{mode}_pro")
    dir_exists(save_path)
    remove_files(save_path)
    if name == "DRIVE":
        img_path = os.path.join(data_path, mode, "image")
        gt_path = os.path.join(data_path, mode, "1st_manual")
        file_list = list(sorted(os.listdir(img_path)))
    img_list = []
    gt_list = []
    img_name = []
    for i, file in enumerate(file_list):
        if name == "DRIVE":
            img_name.append(file[:-4])
            img = Image.open(os.path.join(img_path, file))
            # img = cv2.imread(os.path.join(img_path, file))
            if int(file.split('.')[0]) < 10:
                gt = Image.open(os.path.join(gt_path, '0' + file[0:1] + "_manual1.gif"))
            else:
                if int(file.split('.')[0]) > 20:
                    if int(file.split('.')[0]) - 20 < 10:
                        gt = Image.open(os.path.join(gt_path, '0'+str(int(file[0:2])-20) + "_manual1.gif"))
                    else:
                        gt = Image.open(os.path.join(gt_path, str(int(file[0:2])-20) + "_manual1.gif"))
                else:
                    gt = Image.open(os.path.join(gt_path, file[0:2] + "_manual1.gif"))
            img = Grayscale(1)(img)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            # clahe_img = clahe.apply(img)
            # img_list.append(ToTensor()(clahe_img))
            img_list.append(ToTensor()(img))
            gt_list.append(ToTensor()(gt))
            
    img_list = normalization(img_list)
    if mode == "training":
        img_patch = get_patch(img_list, patch_size, stride)
        gt_patch = get_patch(gt_list, patch_size, stride)
        save_patch(img_patch, save_path, "img_patch", name)
        save_patch(gt_patch, save_path, "gt_patch", name)
    elif mode == "test":
        if name != "CHUAC":
            img_list = get_square(img_list, name)
            gt_list = get_square(gt_list, name)
        save_each_image(img_list, save_path, "img", name, img_name)
        save_each_image(gt_list, save_path, "gt", name, img_name)


def get_square(img_list, name):
    img_s = []
    if name == "DRIVE":
        shape = 592
    elif name == "CHASEDB1":
        shape = 1008
    elif name == "DCA1":
        shape = 320
    _, h, w = img_list[0].shape
    pad = nn.ConstantPad2d((0, shape-w, 0, shape-h), 0)
    for i in range(len(img_list)):
        print(f'before img:{img_list[i].shape}')
        img = pad(img_list[i])
        print(f'pad img:{img.shape}')
        img_s.append(img)

    return img_s


def get_patch(imgs_list, patch_size, stride):
    image_list = []
    _, h, w = imgs_list[0].shape
    pad_h = stride - (h - patch_size) % stride
    pad_w = stride - (w - patch_size) % stride
    for sub1 in imgs_list:
        image = F.pad(sub1, (0, pad_w, 0, pad_h), "constant", 0)
        image = image.unfold(1, patch_size, stride).unfold(
            2, patch_size, stride).permute(1, 2, 0, 3, 4)
        image = image.contiguous().view(
            image.shape[0] * image.shape[1], image.shape[2], patch_size, patch_size)
        for sub2 in image:
            image_list.append(sub2)
    return image_list


def save_patch(imgs_list, path, type, name):
    for i, sub in enumerate(imgs_list):
        with open(file=os.path.join(path, f'{type}_{i}.pkl'), mode='wb') as file:
            pickle.dump(np.array(sub), file)
            print(f'save {name} {type} : {type}_{i}.pkl')


def save_each_image(imgs_list, path, type, name, img_name):
    for i, sub in enumerate(imgs_list):
        with open(file=os.path.join(path, f'{type}_{img_name[i]}.pkl'), mode='wb') as file:
            pickle.dump(np.array(sub), file)
            print(f'save {name} {type} : {type}_{i}.pkl')


def normalization(imgs_list):
    imgs = torch.cat(imgs_list, dim=0)
    mean = torch.mean(imgs)
    std = torch.std(imgs)
    normal_list = []
    for i in imgs_list:
        n = Normalize([mean], [std])(i)
        n = (n - torch.min(n)) / (torch.max(n) - torch.min(n))
        normal_list.append(n)
    return normal_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset_path', default="datasets/DRIVE", type=str,
                        help='the path of dataset',required=True)
    parser.add_argument('-dn', '--dataset_name', default="DRIVE", type=str,
                        help='the name of dataset',choices=['DRIVE','CHASEDB1','STARE','CHUAC','DCA1'],required=True)
    parser.add_argument('-ps', '--patch_size', default=48,
                        help='the size of patch for image partition')
    parser.add_argument('-s', '--stride', default=6,
                        help='the stride of image partition')
    args = parser.parse_args()
    with open('config.yaml', encoding='utf-8') as file:
        # CFG = safe_load(file)  
        yaml = YAML(typ='safe', pure=True)
        CFG = yaml.load(file) # 为列表类型

    # data_process(args.dataset_path, args.dataset_name,
    #              args.patch_size, args.stride, "training")
    data_process(args.dataset_path, args.dataset_name,
                 args.patch_size, args.stride, "test")
