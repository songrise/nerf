# -*- coding : utf-8 -*-
# @FileName  : grad_test.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : 2022-01-26
# @Github    ：https://github.com/songrise
# @Descriptions: To test gradient tape in tensorflow

# %%

import tensorflow as tf
import numpy as np


def get_model():
    """
    simulate nerf model
    """
    input_ = tf.keras.Input(shape=(3,))
    output = tf.keras.layers.Dense(units=4, activation="sigmoid")(input_)
    output = tf.keras.layers.Dense(units=4, activation="sigmoid")(output)
    # simulate the positional MLP

    # sigma
    sigma = tf.keras.layers.Dense(units=1)(output)
    # pred normal
    norm = tf.keras.layers.Dense(units=3)(output)
    # simulate the directional MLP
    output = tf.concat([norm, sigma], axis=1)
    output = tf.keras.layers.Dense(units=4, activation="sigmoid")(output)
    output = tf.keras.layers.Dense(units=1)(output)

    # also output normal for regularization
    model_out = tf.concat([output, sigma, norm], axis=1)
    model = tf.keras.Model(inputs=input_, outputs=model_out)
    return model


model = get_model()
# model.compile()
# model.summary()
gradVar = model.trainable_variables
# random generate data of [N,3]
input_ = tf.random.normal(shape=(10, 3))
input_var = tf.Variable(input_)
gt_out = tf.Variable(np.array([[0.0], [1.0], [4.0], [9.0]]), dtype=tf.float32)
# gt_inter = tf.Variable(np.array([[1.0, 1.0, 1.0]], dtype=np.float32))
# duplicate gt_inter to [N,3]
# gt_inter = np.repeat(gt_inter, 10, axis=0)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for _ in range(5000):
    with tf.GradientTape(persistent=True) as tape:
        # with tf.GradientTape() as tape2:
        out = model(input_)
        # extract the internal value
        pred_out = out[..., 0]
        sigma = out[..., 1]
        pred_normal = out[..., 2:]
        # find the gradient of the internal value
        #!gradient of inter wrt input,
        sigma_grad = tape.gradient(sigma, input_)
        sigma_grad = tf.stop_gradient(sigma_grad)
        # regularization
        r_loss = tf.reduce_mean(tf.square(pred_normal - sigma_grad))
        loss = tf.reduce_sum(tf.square(pred_out - gt_out))
        loss += r_loss

    grad = tape.gradient(loss, gradVar)
    optimizer.apply_gradients(zip(grad, gradVar))

# check if the gradient of internal value is correct
    if _ % 300 == 0:
        # print(inter_grad)
        print(loss)
        print(sigma_grad)

# %%
