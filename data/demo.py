import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataloader.MS2_dataset import DataLoader_MS2
from utils.utils import visualize_disp_as_numpy, visualize_depth_as_numpy, Raw2Celsius
from skimage.color import rgb2gray
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# Enable Matplotlib Interactive Mode for Faster Rendering
plt.ion()

def fill_depth_colorization(imgRgb, imgDepth, alpha=1):
    """
    Implements the fill_depth_colorization method from KITTI Dense Depth.
    """
    imgIsNoise = (imgDepth == 0) | (imgDepth == 10)
    maxImgAbsDepth = np.max(imgDepth[~imgIsNoise])
    imgDepth = imgDepth / 10000.0
    imgDepth[imgDepth > 1] = 1
    H, W = imgDepth.shape
    numPix = H * W
    indsM = np.arange(numPix).reshape(H, W)
    knownValMask = ~imgIsNoise
    grayImg = rgb2gray(imgRgb)
    winRad = 1
    
    rows, cols, vals = [], [], []
    for j in range(W):
        for i in range(H):
            absImgNdx = indsM[i, j]
            gvals = []
            for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
                for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
                    if ii == i and jj == j:
                        continue
                    rows.append(absImgNdx)
                    cols.append(indsM[ii, jj])
                    gvals.append(grayImg[ii, jj])
            
            curVal = grayImg[i, j]
            gvals.append(curVal)
            c_var = np.mean((np.array(gvals) - np.mean(gvals)) ** 2)
            csig = c_var * 0.6
            mgv = np.min((np.array(gvals[:-1]) - curVal) ** 2)
            if csig < (-mgv / np.log(0.01)):
                csig = -mgv / np.log(0.01)
            if csig < 0.000002:
                csig = 0.000002
            
            gvals[:-1] = np.exp(-(np.array(gvals[:-1]) - curVal) ** 2 / csig)
            gvals[:-1] /= np.sum(gvals[:-1])
            vals.extend(-np.array(gvals[:-1]))

            rows.append(absImgNdx)
            cols.append(absImgNdx)
            vals.append(1)
    
    A = csr_matrix((vals, (rows, cols)), shape=(numPix, numPix))
    G = csr_matrix((knownValMask.ravel() * alpha, (np.arange(numPix), np.arange(numPix))), shape=(numPix, numPix))
    
    new_vals = spsolve(A + G, (knownValMask.ravel() * alpha * imgDepth.ravel()))
    denoisedDepthImg = new_vals.reshape(H, W) * maxImgAbsDepth
    return denoisedDepthImg

def save_densified_map(dense_depth, original_depth_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    filename = os.path.basename(original_depth_path)
    save_path = os.path.join(save_dir, filename)

    # Use the visualized version
    vis = visualize_depth_as_numpy(dense_depth)

    # Save with same size as original (no axes, no padding)
    dpi = 100  # dots per inch
    height, width, _ = vis.shape
    figsize = width / dpi, height / dpi

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.Axes(fig, [0, 0, 1, 1])  # full-figure axes
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(vis)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='F:/w1872042_FinalProjectCode/data/MS2dataset')
    parser.add_argument('--seq_name', type=str, default='_2021-08-06-11-23-45', help='sequence name')
    return parser.parse_args()

def main():
    args = parse_args()
    dataset_dir = args.dataset_dir
    seq_name = args.seq_name
    
    modalities = [ 'nir']
    data_formats = ['MonoDepth', 'StereoMatch', 'MultiViewImg']
    
    for modality in modalities:
        for data_format in data_formats:
            print(f"Processing: Modality = {modality}, Data Format = {data_format}")
            dataset = DataLoader_MS2(
                dataset_dir,
                data_split=seq_name,
                data_format=data_format,
                modality=modality,
                sampling_step=1,
                set_length=1,     
                set_interval=1
            )
            demo_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
            print(f'{len(demo_loader)} samples found for evaluation.')
            
            for _, batch in enumerate(tqdm(demo_loader)):
                if data_format in ['MonoDepth', 'StereoMatch', 'MultiViewImg']:
                    # **Fixing Input Image Handling**
                    if modality == 'thr':
                        img = Raw2Celsius(batch["tgt_image"])
                    else:
                        img = batch["tgt_image"].type(torch.uint8).squeeze().cpu().numpy()

                    # Convert Grayscale to RGB if needed
                    if len(img.shape) == 2:
                        img = np.stack([img] * 3, axis=-1)

                    # **Fixing Sparse Depth Handling**
                    sparse_depth = batch["tgt_depth_gt"].squeeze().cpu().numpy()
                    dense_depth = fill_depth_colorization(img, sparse_depth, alpha=1)
                    
                    # Save dense depth maps only for NIR modality
                    if modality == 'nir':
                        dense_gt_path = batch["tgt_depth_gt_path"][0] if isinstance(batch["tgt_depth_gt_path"], list) else batch["tgt_depth_gt_path"]

                        save_densified_map(dense_depth, dense_gt_path, './nir_dense_depth_maps')

                    # **Matplotlib Optimization**
                    plt.clf()

                    plt.subplot(2, 2, 1)
                    plt.imshow(img)
                    plt.title("Input Image")

                    plt.subplot(2, 2, 2)
                    plt.imshow(visualize_depth_as_numpy(sparse_depth))
                    plt.title("Sparse Depth Map")

                    plt.subplot(2, 2, 3)
                    plt.imshow(visualize_depth_as_numpy(dense_depth))
                    plt.title("Densified Depth Map (KITTI Method)")

                    plt.pause(0.1)

if __name__ == '__main__':
    main()