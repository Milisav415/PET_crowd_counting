import argparse
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as standard_transforms

import util.misc as utils
from models import build_model

from video_panorama_lib import create_panorama  # Imported if needed later


def get_args_parser():
    parser = argparse.ArgumentParser('Set Point Query Transformer', add_help=False)
    # Model parameters
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned', 'fourier'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=512, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    # Loss parameters
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="SmoothL1 point coefficient in the matching cost")
    parser.add_argument('--ce_loss_coef', default=1.0, type=float)
    parser.add_argument('--point_loss_coef', default=5.0, type=float)
    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")
    # Dataset parameters
    parser.add_argument('--dataset_file', default="SHA")
    parser.add_argument('--data_path', default="./data/ShanghaiTech/PartA", type=str)
    # Misc parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--vis_dir', default="")
    parser.add_argument('--num_workers', default=2, type=int)
    # Grid segmentation parameters
    parser.add_argument('--grid_cols', default=2, type=int,
                        help="Number of columns in the segmentation grid")
    parser.add_argument('--grid_rows', default=2, type=int,
                        help="Number of rows in the segmentation grid")
    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def visualization(samples, pred, folder_path, segment_name, split_map=None):
    """
    Visualize predictions on a segment image and save the visualization in folder_path.
    The output filename includes the segment name and its prediction count.
    """
    pil_to_tensor = standard_transforms.ToTensor()
    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])

    images = samples.tensors
    masks = samples.mask

    for idx in range(images.shape[0]):
        sample = restore_transform(images[idx])
        sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
        sample_vis = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()

        # Draw predictions (green circles)
        size = 3
        for p in pred[idx]:
            sample_vis = cv2.circle(sample_vis, (int(p[1]), int(p[0])), size, (0, 255, 0), -1)

        # Draw split map if provided
        if split_map is not None:
            imgH, imgW = sample_vis.shape[:2]
            split_map_img = (split_map * 255).astype(np.uint8)
            split_map_img = cv2.applyColorMap(split_map_img, cv2.COLORMAP_JET)
            split_map_img = cv2.resize(split_map_img, (imgW, imgH), interpolation=cv2.INTER_NEAREST)
            sample_vis = cv2.addWeighted(sample_vis, 0.9, split_map_img, 0.1, 0)

        # Eliminate invalid area using the mask
        imgH, imgW = masks.shape[-2:]
        valid_area = torch.where(~masks[idx])
        valid_h, valid_w = valid_area[0][-1], valid_area[1][-1]
        sample_vis = sample_vis[:valid_h + 1, :valid_w + 1]

        # Build the filename with segment name and its prediction count
        segment_pred_count = len(pred[idx])
        segment_filename = f"{segment_name}_pred{segment_pred_count}.jpg"
        img_save_path = os.path.join(folder_path, segment_filename)
        cv2.imwrite(img_save_path, sample_vis)
        print('Segment saved to', img_save_path)


@torch.no_grad()
def evaluate_segment(model, segment, device, folder_path, segment_name):
    """
    Process a single segment: apply transforms, run inference, visualize predictions.
    Returns the number of predictions (points) for that segment.
    """
    model.eval()

    if not isinstance(segment, Image.Image):
        segment = Image.fromarray(segment)

    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(segment)
    samples = utils.nested_tensor_from_tensor_list([img_tensor])
    samples = samples.to(device)
    img_h, img_w = samples.tensors.shape[-2:]

    outputs = model(samples, test=True)
    raw_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)
    outputs_points = outputs['pred_points'][0]
    pred_count = len(outputs_points)
    print(f"Prediction for {segment_name}: {pred_count} points detected")

    points = [[point[0] * img_h, point[1] * img_w] for point in outputs_points]
    split_map = None
    if 'split_map_raw' in outputs:
        split_map = (outputs['split_map_raw'][0].detach().cpu().squeeze(0) > 0.5).float().numpy()
    visualization(samples, [points], folder_path, segment_name, split_map=split_map)
    return pred_count


def segment_image(image, grid_cols, grid_rows):
    """
    Segments an image into a grid with grid_cols columns and grid_rows rows.
    Returns a list of image segments (as numpy arrays).
    """
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    height, width = image_np.shape[:2]
    cell_width = width // grid_cols
    cell_height = height // grid_rows
    segments = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            x1 = col * cell_width
            y1 = row * cell_height
            x2 = width if col == grid_cols - 1 else (col + 1) * cell_width
            y2 = height if row == grid_rows - 1 else (row + 1) * cell_height
            segment = image_np[y1:y2, x1:x2]
            segments.append(segment)
    return segments


@torch.no_grad()
def evaluate_single_image(model, img_path, device, vis_dir=None):
    model.eval()

    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)

    # load image
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # transform image
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img)
    img = torch.Tensor(img)
    samples = utils.nested_tensor_from_tensor_list([img])
    samples = samples.to(device)
    img_h, img_w = samples.tensors.shape[-2:]

    # inference
    outputs = model(samples, test=True)
    raw_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)
    outputs_scores = raw_scores[:, :, 1][0]
    outputs_points = outputs['pred_points'][0]
    print('prediction: ', len(outputs_scores))

    # visualize predictions
    if vis_dir:
        points = [[point[0] * img_h, point[1] * img_w] for point in outputs_points]  # recover to actual points
        split_map = (outputs['split_map_raw'][0].detach().cpu().squeeze(0) > 0.5).float().numpy()
        visualization(samples, [points], vis_dir, img_path, split_map=split_map)


def main(args):
    # Build model
    device = torch.device(args.device)
    model, criterion = build_model(args)
    model.to(device)

    checkpoint = torch.load(args.resume, map_location='cpu')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise KeyError("No 'model' or 'state_dict' key found in the checkpoint.")

    # Create a single visualization folder based on the input image's basename
    base_name = os.path.splitext(os.path.basename(args.img_path))[0]
    folder_path = os.path.join(args.vis_dir, base_name)
    os.makedirs(folder_path, exist_ok=True)

    # Load the full image and segment it
    full_image = Image.open(args.img_path).convert('RGB')
    segments = segment_image(full_image, args.grid_cols, args.grid_rows)
    print(f"Image segmented into {len(segments)} pieces.")

    total_preds = 0
    for idx, seg in enumerate(segments):
        segment_name = f"{base_name}_segment_{idx}"
        pred_count = evaluate_segment(model, seg, device, folder_path, segment_name)
        total_preds += pred_count

    # Write a summary text file with the total predictions (dots)
    summary_file = os.path.join(folder_path, "summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"Total predictions (dots): {total_preds}\n")
    print(f"Summary written to {summary_file}")


if __name__ == '__main__':

    # Instead of parsing command line arguments, we manually create an args object.
    class Args:
        pass

    args = Args()
    # Model parameters
    args.backbone = 'vgg19_bn'
    args.position_embedding = 'sine'
    args.dec_layers = 2
    args.dim_feedforward = 512
    args.hidden_dim = 256
    args.dropout = 0.0
    args.nheads = 8
    # Loss parameters
    args.set_cost_class = 1.0
    args.set_cost_point = 0.05
    args.ce_loss_coef = 1.0
    args.point_loss_coef = 5.0
    args.eos_coef = 0.5
    # Misc parameters
    args.device = 'cpu'  # change to 'cuda' if available
    args.seed = 42
    args.num_workers = 2
    # Distributed training parameters
    args.world_size = 1
    args.dist_url = 'env://'
    # Input paths and grid segmentation parameters
    args.img_path = r'C:\Users\jm190\Desktop\Deep Learning\7\DJI_0061.tif'  # update as needed
    args.resume = r'C:\Users\jm190\PycharmProjects\PET_cc\pretrained\JHU_Crowd.pth'  # update as needed
    args.vis_dir = r'C:\Users\jm190\PycharmProjects\PET_cc\visualization'
    args.grid_cols = 3  # number of columns in the segmentation grid
    args.grid_rows = 3  # number of rows in the segmentation grid
    # Run the main function
    main(args) # here we go
