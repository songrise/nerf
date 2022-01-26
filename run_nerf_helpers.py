# %%
import os
import sys
import tensorflow as tf
import numpy as np
import imageio
import json


# Misc utils

def img2mse(x, y): return tf.reduce_mean(tf.square(x - y))


def mse2psnr(x): return -10.*tf.log(x)/tf.log(10.)


def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)


# Positional encoding

class Embedder:
    #! Re should be the \gamma for positional encoding?

    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**tf.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = tf.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return tf.concat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):

    if i == -1:
        # return itself without any embedding
        return tf.identity, 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [tf.math.sin, tf.math.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim


class SphericalBasis(tf.keras.Model):
    def __init__(self, n_order=8, inputch_views=27, **kwargs):
        """
        Args:
            n_order: the order of the basis function
        """
        super(SphericalBasis, self).__init__()
        self.n_order = n_order
        self.basis = []
        for _ in range(n_order):
            if _ <= n_order//4:
                self.basis.append(self.create_basis(
                    inputch_views=inputch_views, width=8))
            elif _ <= n_order//2:
                self.basis.append(self.create_basis(
                    inputch_views=inputch_views, width=16))
            else:
                self.basis.append(self.create_basis(
                    inputch_views=inputch_views, width=32))

    def call(self, inputs,  coeff=None):
        """
        Args:
            inputs: the embedded view direction
            coeff: the coefficients of the basis function [batch_size, n_order, 1]
        Returns:
            rgb: (None, 3) predicted rgb in the range [-1, 1]
        """
        if coeff is None:
            coeff = tf.ones_like(inputs)
            print("coeff is empty")

        # duplicate the coeff for 3 channels
        coeff = tf.tile(tf.expand_dims(coeff, axis=2), [1, 1, 3])

        x = inputs
        rgb = self.basis[0](x)

        for i in range(1, self.n_order):
            # linear combination of basis functions
            rgb += coeff[:, i]*self.basis[i](x)
        # rgb in [-1, 1] since the view-dependent effect maybe black
        # rgb = tf.keras.activations.(rgb)
        return rgb

    def create_basis(self, width=8, inputch_views=27):
        """
        Create the basis function
        """
        basis = tf.keras.Sequential([
            tf.keras.layers.Dense(width, activation=tf.keras.activations.relu,
                                  input_shape=(inputch_views,)),
            tf.keras.layers.Dense(width, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(3, activation=None)
        ])
        return basis


# Model architecture

# def init_nerf_model(D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
#     #!Re see p.18 of the paper for the model architecture
#     relu = tf.keras.layers.ReLU()
#     def dense(W, act=relu): return tf.keras.layers.Dense(W, activation=act)

#     print('MODEL', input_ch, input_ch_views, type(
#         input_ch), type(input_ch_views), use_viewdirs)
#     input_ch = int(input_ch)
#     input_ch_views = int(input_ch_views)

#     inputs = tf.keras.Input(shape=(input_ch + input_ch_views))
#     inputs_pts, inputs_views = tf.split(inputs, [input_ch, input_ch_views], -1)
#     inputs_pts.set_shape([None, input_ch])
#     inputs_views.set_shape([None, input_ch_views])

#     print(inputs.shape, inputs_pts.shape, inputs_views.shape)
#     outputs = inputs_pts
#     for i in range(D):
#         outputs = dense(W)(outputs)
#         if i in skips:
#             outputs = tf.concat([inputs_pts, outputs], -1)

#     if use_viewdirs:
#         # !Re alpha is view-independent, if not used, then it simulates Lambertian。
#         alpha_out = dense(1, act=None)(outputs)  # !Re shouldn't this be relu?
#         bottleneck = dense(256, act=None)(outputs)
#         # !Re input view direction here
#         inputs_viewdirs = tf.concat(
#             [bottleneck, inputs_views], -1)  # concat viewdirs
#         outputs = inputs_viewdirs
#         # The supplement to the paper states there are 4 hidden layers here, but this is an error since
#         # the experiments were actually run with 1 hidden layer, so we will leave it as 1.
#         # ! Re in CVPR paper, there is only one layer.
#         for i in range(1):
#             outputs = dense(W//2)(outputs)
#         # !Re: rgb, shoudn't this be sigmoid?
#         outputs = dense(3, act=None)(outputs)
#         outputs = tf.concat([outputs, alpha_out], -1)  # !Re rgb+a
#     else:
#         outputs = dense(output_ch, act=None)(outputs)

#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model

# # modified architecture
# def init_nerf_model(D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
#     #!Re see p.18 of the paper for the model architecture
#     relu = tf.keras.layers.ReLU()
#     def dense(W, act=relu): return tf.keras.layers.Dense(W, activation=act)

#     print('MODEL', input_ch, input_ch_views, type(
#         input_ch), type(input_ch_views), use_viewdirs)
#     input_ch = int(input_ch)
#     input_ch_views = int(input_ch_views)

#     inputs = tf.keras.Input(shape=(input_ch + input_ch_views))
#     inputs_pts, inputs_views = tf.split(inputs, [input_ch, input_ch_views], -1)
#     inputs_pts.set_shape([None, input_ch])
#     inputs_views.set_shape([None, input_ch_views])

#     print(inputs.shape, inputs_pts.shape, inputs_views.shape)
#     outputs = inputs_pts
#     for i in range(D):
#         outputs = dense(W)(outputs)
#         if i in skips:
#             outputs = tf.concat([inputs_pts, outputs], -1)
#     n_order = 8  # neural expansion order
#     if use_viewdirs:
#         #! Re modified this following NeX and refNerf
#         alpha_out = dense(1, act=None)(outputs)
#         diffuse_out = dense(3, act="sigmoid")(outputs)
#         bottleneck = dense(256, act=None)(outputs)

#         inputs_viewdirs = tf.concat(
#             [bottleneck, inputs_views], -1)  # concat viewdirs
#         outputs = inputs_viewdirs
#         # The supplement to the paper states there are 4 hidden layers here, but this is an error since
#         # the experiments were actually run with 1 hidden layer, so we will leave it as 1.
#         for i in range(1):  # !Re: Modified for 4 layers in order to get better view-dependent results
#             outputs = dense(W//2)(outputs)

#         outputs = dense(n_order, act=None)(outputs)  # coeff
#         basis = SphericalBasis(n_order, input_ch_views)
#         outputs = basis(inputs_views, coeff=outputs)  # specular
#         outputs = diffuse_out + outputs  # diffuse + specular
#         outputs = tf.concat([outputs, alpha_out], -1)  # !Re rgb+a
#     else:
#         outputs = dense(output_ch, act=None)(outputs)

#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model

# %%


# Misc utils

def img2mse(x, y): return tf.reduce_mean(tf.square(x - y))


def mse2psnr(x): return -10.*tf.log(x)/tf.log(10.)


def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)


# Positional encoding

class Embedder:
    #! Re should be the \gamma for positional encoding?

    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**tf.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = tf.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return tf.concat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):

    if i == -1:
        # return itself without any embedding
        return tf.identity, 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [tf.math.sin, tf.math.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim


class SphericalBasis(tf.keras.Model):
    def __init__(self, n_order=8, inputch_views=27, **kwargs):
        """
        Args:
            n_order: the order of the basis function
        """
        super(SphericalBasis, self).__init__()
        self.n_order = n_order
        self.basis = []
        for _ in range(n_order):
            if _ <= n_order//4:
                self.basis.append(self.create_basis(
                    inputch_views=inputch_views, width=8))
            elif _ <= n_order//2:
                self.basis.append(self.create_basis(
                    inputch_views=inputch_views, width=32))
            else:
                self.basis.append(self.create_basis(
                    inputch_views=inputch_views, width=64))

    def call(self, inputs,  coeff=None):
        """
        Args:
            inputs: the embedded view direction
            coeff: the coefficients of the basis function [batch_size, n_order, 1]
        Returns:
            rgb: (None, 3) predicted rgb in the range [-1, 1]
        """
        if coeff is None:
            coeff = tf.ones_like(inputs)
            print("coeff is empty")

        # duplicate the coeff for 3 channels
        coeff = tf.tile(tf.expand_dims(coeff, axis=2), [1, 1, 3])

        x = inputs
        rgb = self.basis[0](x)

        for i in range(1, self.n_order):
            # linear combination of basis functions
            rgb += coeff[:, i]*self.basis[i](x)
        # rgb in [-1, 1] since the view-dependent effect maybe black
        # rgb = tf.keras.activations.(rgb)
        return rgb

    def create_basis(self, width=8, inputch_views=27):
        """
        Create the basis function
        """
        basis = tf.keras.Sequential([
            tf.keras.layers.Dense(width, activation=tf.keras.activations.elu,
                                  input_shape=(inputch_views,)),
            tf.keras.layers.Dense(width, activation=tf.keras.activations.elu),
            tf.keras.layers.Dense(3, activation=None)
        ])
        return basis


# Model architecture

# def init_nerf_model(D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
#     #!Re see p.18 of the paper for the model architecture
#     relu = tf.keras.layers.ReLU()
#     def dense(W, act=relu): return tf.keras.layers.Dense(W, activation=act)

#     print('MODEL', input_ch, input_ch_views, type(
#         input_ch), type(input_ch_views), use_viewdirs)
#     input_ch = int(input_ch)
#     input_ch_views = int(input_ch_views)

#     inputs = tf.keras.Input(shape=(input_ch + input_ch_views))
#     inputs_pts, inputs_views = tf.split(inputs, [input_ch, input_ch_views], -1)
#     inputs_pts.set_shape([None, input_ch])
#     inputs_views.set_shape([None, input_ch_views])

#     print(inputs.shape, inputs_pts.shape, inputs_views.shape)
#     outputs = inputs_pts
#     for i in range(D):
#         outputs = dense(W)(outputs)
#         if i in skips:
#             outputs = tf.concat([inputs_pts, outputs], -1)

#     if use_viewdirs:
#         # !Re alpha is view-independent, if not used, then it simulates Lambertian。
#         alpha_out = dense(1, act=None)(outputs)  # !Re shouldn't this be relu?
#         bottleneck = dense(256, act=None)(outputs)
#         # !Re input view direction here
#         inputs_viewdirs = tf.concat(
#             [bottleneck, inputs_views], -1)  # concat viewdirs
#         outputs = inputs_viewdirs
#         # The supplement to the paper states there are 4 hidden layers here, but this is an error since
#         # the experiments were actually run with 1 hidden layer, so we will leave it as 1.
#         # ! Re in CVPR paper, there is only one layer.
#         for i in range(1):
#             outputs = dense(W//2)(outputs)
#         # !Re: rgb, shoudn't this be sigmoid?
#         outputs = dense(3, act=None)(outputs)
#         outputs = tf.concat([outputs, alpha_out], -1)  # !Re rgb+a
#     else:
#         outputs = dense(output_ch, act=None)(outputs)

#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model



# modified architecture v2
def init_nerf_model(D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
    #!Re see p.18 of the paper for the model architecture
    relu = tf.keras.layers.ReLU()
    def dense(W, act=relu): return tf.keras.layers.Dense(W, activation=act)

    print('MODEL', input_ch, input_ch_views, type(
        input_ch), type(input_ch_views), use_viewdirs)
    input_ch = int(input_ch)
    input_ch_views = int(input_ch_views)

    inputs = tf.keras.Input(shape=(input_ch + input_ch_views))
    inputs_pts, inputs_views = tf.split(inputs, [input_ch, input_ch_views], -1)
    inputs_pts.set_shape([None, input_ch])
    inputs_views.set_shape([None, input_ch_views])

    print(inputs.shape, inputs_pts.shape, inputs_views.shape)
    outputs = inputs_pts
    for i in range(D):
        outputs = dense(W)(outputs)
        if i in skips:
            outputs = tf.concat([inputs_pts, outputs], -1)
    n_order = 8  # neural expansion order
    if use_viewdirs:
        #! Re modified this following NeX and refNerf
        alpha_out = dense(1, act=None)(outputs)
        diffuse_out = dense(3, act=None)(outputs)
        bottleneck = dense(256, act=None)(outputs)

        inputs_viewdirs = tf.concat(
            [bottleneck, inputs_views], -1)  # concat viewdirs
        outputs = inputs_viewdirs
        # The supplement to the paper states there are 4 hidden layers here, but this is an error since
        # the experiments were actually run with 1 hidden layer, so we will leave it as 1.
        outputs = dense(W//2)(outputs)
        isOutline = dense(1, act='sigmoid')(outputs)
        isOutline = tf.pow(isOutline, 6)
        # threshold = tf.Variable(0.5, dtype=tf.float32)
        # isOutline = tf.cast(isOutline > 0.5, tf.float32)

        # outputs = dense(n_order, act=None)(outputs)  # coeff
        # basis = SphericalBasis(n_order, input_ch_views)
        # specular_out = basis(inputs_views, coeff=outputs)  # specular
        # pure black
        # specular_out = tf.constant([0, 0, 0], dtype=tf.float32)
        # if it is not outline, not affected
        outputs = (1-isOutline)*diffuse_out
        # specular_out  # diffuse + specular
        outputs = tf.concat([outputs, alpha_out], -1)  # !Re rgb+a
    else:
        outputs = dense(output_ch, act=None)(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Ray helpers


def get_rays(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    #!Re: c2w is the camera to world matrix, which is the inverse of the world to camera matrix.
    #!Re: https://en.wikipedia.org/wiki/Pinhole_camera_model
    i, j = tf.meshgrid(tf.range(W, dtype=tf.float32),
                       tf.range(H, dtype=tf.float32), indexing='xy')
    #!Re multiply by .5 is just dividing by 2.
    #!Re i,j is the mapped into [-W/2, W/2], see pinhole camera model for intuition.
    #!Re dirs are the normalized ray directions in camera space. the z axis is fixed.
    dirs = tf.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -tf.ones_like(i)], -1)
    #!Re: c2w[:3, :3] is rotation part of the camera to world matrix.
    rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    #! Re: c2w[:3, -1] is the translation part.
    rays_o = tf.broadcast_to(c2w[:3, -1], tf.shape(rays_d))
    return rays_o, rays_d

#!Re see https://github.com/bmild/nerf/issues/92


def get_rays_np(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    if None:
        return get_rays_np_sub_pix(H, W, focal, c2w)

    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def get_rays_np_sub_pix(H, W, focal, c2w):
    """
    Get ray origins, directions from a pinhole camera.
    Sample 4 rays for a single pixel
    """
    i, j = np.meshgrid(np.arange(2*W, dtype=np.float32),
                       np.arange(2*H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W)/(focal), -(j-H) /
                     (focal), -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # just divide them by 2 since the rays are centered wrt the principal ray
    rays_d /= np.array([2, 2, 1])
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


a = get_rays_np_sub_pix(2, 2, 0.5, np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """Normalized device coordinate rays.

    Space such that the canvas is a cube with sides [-1, 1] in each axis.

    Args:
      H: int. Height in pixels.
      W: int. Width in pixels.
      focal: float. Focal length of pinhole camera.
      near: float or array of shape[batch_size]. Near depth bound for the scene.
      rays_o: array of shape [batch_size, 3]. Camera origin.
      rays_d: array of shape [batch_size, 3]. Ray direction.

    Returns:
      rays_o: array of shape [batch_size, 3]. Camera origin in NDC.
      rays_d: array of shape [batch_size, 3]. Ray direction in NDC.
    """
    # Shift ray origins to near plane
    #!Re todo check this later.
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1./(W/(2.*focal)) * \
        (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(H/(2.*focal)) * \
        (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = tf.stack([o0, o1, o2], -1)
    rays_d = tf.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Hierarchical sampling helper

def sample_pdf(bins, weights, N_samples, det=False):

    # Get pdf
    weights += 1e-5  # prevent nans
    #! Re normalize
    pdf = weights / tf.reduce_sum(weights, -1, keepdims=True)
    cdf = tf.cumsum(pdf, -1)
    cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        #! Re if deterministic, then take uniform samples
        u = tf.linspace(0., 1., N_samples)
        u = tf.broadcast_to(u, list(cdf.shape[:-1]) + [N_samples])
    else:
        #!Re else take samples from uniform distribution
        u = tf.random.uniform(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    #!Re index of the bin
    inds = tf.searchsorted(cdf, u, side='right')
    below = tf.maximum(0, inds-1)
    above = tf.minimum(cdf.shape[-1]-1, inds)
    inds_g = tf.stack([below, above], -1)
    cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples


# %%
if __name__ == "__main__":
    rays = [get_rays_np_sub_pix(4, 4, 0.5, np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])) for _ in range(16)]
    rays = np.stack(rays, 0)
# %%

# Ray helpers


def get_rays(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    #!Re: c2w is the camera to world matrix, which is the inverse of the world to camera matrix.
    #!Re: https://en.wikipedia.org/wiki/Pinhole_camera_model
    i, j = tf.meshgrid(tf.range(W, dtype=tf.float32),
                       tf.range(H, dtype=tf.float32), indexing='xy')
    #!Re multiply by .5 is just dividing by 2.
    #!Re i,j is the mapped into [-W/2, W/2], see pinhole camera model for intuition.
    #!Re dirs are the normalized ray directions in camera space. the z axis is fixed.
    dirs = tf.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -tf.ones_like(i)], -1)
    #!Re: c2w[:3, :3] is rotation part of the camera to world matrix.
    rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    #! Re: c2w[:3, -1] is the translation part.
    rays_o = tf.broadcast_to(c2w[:3, -1], tf.shape(rays_d))
    return rays_o, rays_d

#!Re see https://github.com/bmild/nerf/issues/92


def get_rays_np(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    if None:
        return get_rays_np_sub_pix(H, W, focal, c2w)

    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def get_rays_np_sub_pix(H, W, focal, c2w):
    """
    Get ray origins, directions from a pinhole camera.
    Sample 4 rays for a single pixel
    """
    i, j = np.meshgrid(np.arange(2*W, dtype=np.float32),
                       np.arange(2*H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W)/(focal), -(j-H) /
                     (focal), -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # just divide them by 2 since the rays are centered wrt the principal ray
    rays_d /= np.array([2, 2, 1])
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


a = get_rays_np_sub_pix(2, 2, 0.5, np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """Normalized device coordinate rays.

    Space such that the canvas is a cube with sides [-1, 1] in each axis.

    Args:
      H: int. Height in pixels.
      W: int. Width in pixels.
      focal: float. Focal length of pinhole camera.
      near: float or array of shape[batch_size]. Near depth bound for the scene.
      rays_o: array of shape [batch_size, 3]. Camera origin.
      rays_d: array of shape [batch_size, 3]. Ray direction.

    Returns:
      rays_o: array of shape [batch_size, 3]. Camera origin in NDC.
      rays_d: array of shape [batch_size, 3]. Ray direction in NDC.
    """
    # Shift ray origins to near plane
    #!Re todo check this later.
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1./(W/(2.*focal)) * \
        (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(H/(2.*focal)) * \
        (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = tf.stack([o0, o1, o2], -1)
    rays_d = tf.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Hierarchical sampling helper

def sample_pdf(bins, weights, N_samples, det=False):

    # Get pdf
    weights += 1e-5  # prevent nans
    #! Re normalize
    pdf = weights / tf.reduce_sum(weights, -1, keepdims=True)
    cdf = tf.cumsum(pdf, -1)
    cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        #! Re if deterministic, then take uniform samples
        u = tf.linspace(0., 1., N_samples)
        u = tf.broadcast_to(u, list(cdf.shape[:-1]) + [N_samples])
    else:
        #!Re else take samples from uniform distribution
        u = tf.random.uniform(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    #!Re index of the bin
    inds = tf.searchsorted(cdf, u, side='right')
    below = tf.maximum(0, inds-1)
    above = tf.minimum(cdf.shape[-1]-1, inds)
    inds_g = tf.stack([below, above], -1)
    cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples


# %%
if __name__ == "__main__":
    rays = [get_rays_np_sub_pix(4, 4, 0.5, np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])) for _ in range(16)]
    rays = np.stack(rays, 0)
# %%
