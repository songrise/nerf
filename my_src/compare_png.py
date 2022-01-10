# -*- coding : utf-8 -*-
# @FileName  : compare_png.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : 2022-01-02
# @Github    ：https://github.com/songrise
# @Descriptions: compare the difference between two png files

# %%
import numpy as np
import imageio
import os
# my data
fname = "0000_shot.png"
# official data
# fname = "r_0.png"
fname = os.path.join(fname)
imgs = imageio.imread(fname)
# keep all 4 channels (RGBA)
imgs = (np.array(imgs) / 255.).astype(np.float32)

# %%
print(imgs.shape)
# the first pixel
print(imgs[0, 0])
# %%center
print(imgs[400, 400])
# %%
for i in range(10):
    print(imgs[i,0])
# %%
