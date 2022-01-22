# -*- coding : utf-8 -*-
# @FileName  : basis.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : 2022-01-21
# @Github    ：https://github.com/songrise
# @Descriptions: Neural spherical basis function implemented in TensorFlow 1.x
# %%
from mimetypes import init
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
    def __init__(self, n_order=8):
        """
        Args:
            n_order: the order of the basis function
        """
        super(SphericalBasis, self).__init__()
        self.n_order = n_order
        self.basis = []
        for _ in range(n_order):
            if _ < n_order//4:
                self.basis.append(self.create_basis(width=4))
            elif _ < n_order//2:
                self.basis.append(self.create_basis(width=16))
            else:
                self.basis.append(self.create_basis(width=32))

    def call(self, inputs,  coeff=None):
        """
        Args:
            inputs: the embedded view direction
            coeff: the coefficients of the basis function
        Returns:
            rgb: (None, 3) predicted rgb in the range [-1, 1]
        """
        if coeff is None:
            coeff = tf.ones_like(inputs)
            print("coeff is empty")

        x = inputs
        rgb = self.basis[0](x)
        print(rgb.shape)
        print(coeff.shape)

        for i in range(1, self.n_order):
            # linear combination of basis functions
            rgb += coeff[..., i]*self.basis[i](x)
        # rgb in [-1, 1] since the view-dependent effect maybe black
        rgb = tf.keras.activations.tanh(rgb)
        return rgb

    def create_basis(self, width=4, n_order=8):
        """
        Create the basis function
        """
        basis = tf.keras.Sequential([
            tf.keras.layers.Dense(width, activation='relu',
                                  input_shape=(n_order,)),
            tf.keras.layers.Dense(width, activation='relu'),
            tf.keras.layers.Dense(3, activation='sigmoid')
        ])
        return basis


# ===========Experiment with reconstructing radiance from several samples=================


if __name__ == "__main__":
    model = SphericalBasis(n_order=8)
    model.build(input_shape=(None, 8))
    model.summary()

# %%
