# -*- coding : utf-8 -*-
# @FileName  : basis.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : 2022-01-21
# @Github    ：https://github.com/songrise
# @Descriptions: Neural spherical basis function implemented in TensorFlow 1.x
# %%

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# class Basis(tf.keras.Model):
#     """
#     Neural spherical basis function define over a hemisphere
#     it map the embedded view direction to a real-valued intensity.
#     """

#     def __init__(self,  width=16, n_hidden=2):
#         """
#         Args:

#         """
#         super(Basis, self).__init__()
#         # self.input_dim = input_dim
#         self.n_hidden = n_hidden
#         # self.input_layer = tf.keras.layers.Input(shape=(input_dim,))
#         self.hidden_layers = []
#         for _ in range(n_hidden):
#             self.hidden_layers.append(
#                 tf.keras.layers.Dense(width, activation='relu'))
#         self.rgb_intensity = tf.keras.layers.Dense(3, activation='sigmoid')

#     def call(self, inputs, training=False):
#         """
#         Map the embedded coefficients to a real-valued intensity for rgb
#         """

#         x = inputs

#         for layer in self.hidden_layers:
#             x = layer(x)
#         x = self.rgb_intensity(x)
#         return x

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
            elif _ < n_order-1:
                self.basis.append(self.create_basis(
                    inputch_views=inputch_views, width=64))
            else:
                self.basis.append(self.create_basis(
                    inputch_views=inputch_views, width=128))

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
            tf.keras.layers.Dense(
                width, activation=tf.keras.activations.elu),
            tf.keras.layers.Dense(3, activation=None)
        ])
        return basis


# ===========Experiment with reconstructing radiance from several samples=================

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
        diffuse_out = dense(3, act="sigmoid")(outputs)
        bottleneck = dense(256, act=None)(outputs)

        inputs_viewdirs = tf.concat(
            [bottleneck, inputs_views], -1)  # concat viewdirs
        outputs = inputs_viewdirs
        # The supplement to the paper states there are 4 hidden layers here, but this is an error since
        # the experiments were actually run with 1 hidden layer, so we will leave it as 1.
        for i in range(1):  # !Re: Modified for 4 layers in order to get better view-dependent results
            outputs = dense(W//2)(outputs)

        outputs = dense(n_order, act=None)(outputs)  # coeff
        basis = SphericalBasis(n_order, input_ch_views)
        outputs = basis(inputs_views, coeff=outputs)  # specular
        outputs = diffuse_out + outputs  # diffuse + specular
        outputs = tf.concat([outputs, alpha_out], -1)  # !Re rgb+a
    else:
        outputs = dense(output_ch, act=None)(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == "__main__":
    model = SphericalBasis(n_order=8)
    model.build(input_shape=(None, 8))
    model.summary()
    nerf = init_nerf_model()
    nerf.summary()
# modified architecture


# %%
