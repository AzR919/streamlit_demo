import streamlit as st
import numpy as np
import tifffile
from io import BytesIO
import os

import torch

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from plotting_utils import *

# def plot_1_image(image, cmap='gray', path=None, id=None, show=False):
#     """
#     Plot given 2D+T image as an animation
#     @inputs:
#         image: numpy array of shape [T, H, W], dtype = float
#         cmap: color pellete to use
#         path: if not None, will save on that path
#         id: optional identifier to save the image with
#         show: if True, will show the animated image (will pause the execution)
#     """

#     fig = plt.figure()
    
#     def init():
#         return plt.imshow(image[0],cmap=cmap,animated=True) # start with 0th frame

#     def update(i):
#         return plt.imshow(image[i],cmap=cmap,animated=True) # ith frame for ith timestamp

#     ani = animation.FuncAnimation(fig, update, init_func = init, frames = image.shape[0],
#                                     interval = 200, repeat_delay=2000)
    
#     if path != None:
#         print(f"Saving plot at {path}")
#         save_path = os.path.join(path, f'Image_{id}_video.mp4')
#         print(f"Saving video at {save_path}")
#         ani.save(save_path, writer = 'ffmpeg')

#     plt.close()

cutout_shape = (32, 256, 256)

def device_run(model, noisy_im, device):

    model.to(device)
    noisy_im.to(device)

    return model(noisy_im).cpu().detach().numpy()

def run_model(noisy_im):

    m_path = "/tmp/models/CNNT_Microscopy_DN_2DT_B_16_32_64_num_4_head_8_T_16_H__128_160_192__W__128_160_192__Pytorch_1.11.0+cu113__07-26-2022_T09-03-16_epoch-500_real.pts"

    model = torch.jit.load(m_path)
    model.eval()

    noisy_max = 228.0
    noisy_min = 0.0

    noisy_im = (noisy_im - noisy_min) / (noisy_max - noisy_min)

    T, H, W = noisy_im.shape

    noisy_im = noisy_im[:cutout_shape[0], :cutout_shape[1], :cutout_shape[2]]

    noisy_im = torch.from_numpy(noisy_im.astype(np.float32))

    T, H, W = noisy_im.shape

    noisy_im = noisy_im.reshape(1, T, 1, H, W)

    try:
        st.write("Running on GPU")
        clean_pred = device_run(model, noisy_im, 'cuda').reshape(T, H, W)
    except:
        st.write("Failed on GPU, Running on CPU")
        clean_pred = device_run(model, noisy_im, 'cpu').reshape(T, H, W)

    clean_pred = np.clip(clean_pred, 0, 1)
    st.write("Preparing plots")

    plot_2_images([noisy_im, clean_pred], path="./", id=1, show=False)

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = BytesIO(uploaded_file.read()) 
    
    noisy_im = np.array(tifffile.imread(bytes_data))

    st.write("Cutout shape is")
    st.write(cutout_shape)

    run_model(noisy_im)

    st.video("./Image_1_video.mp4")