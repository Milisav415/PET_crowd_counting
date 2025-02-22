import torch.utils.data
import torchvision

from .SHA import build as build_sha
from .PETCrowdDataset import build as build_petcrowd

data_path = {
    'SHA': './data/ShanghaiTech/part_A/',
    'custom_dataset': r'C:\Users\jm190\Desktop\jhu_crowd_v2.0',  # path to your custom dataset folder
}

def build_dataset(image_set, args):
    args.data_path = data_path[args.dataset_file]
    if args.dataset_file == 'SHA':
        return build_sha(image_set, args)
    elif args.dataset_file == 'custom_dataset':
        return build_petcrowd(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
