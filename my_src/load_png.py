# %%
import numpy as np
import imageio
import os
# my data
fname = "0200_shot.png"
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

# %%
