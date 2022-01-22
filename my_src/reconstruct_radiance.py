# -*- coding : utf-8 -*-
# @FileName  : reconstruct_radiance.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : 2022-01-18
# @Github    ：https://github.com/songrise
# @Descriptions: Experiment with reconstructing radiance from several samples
# %%
import numpy as np
import keras
from matplotlib import pyplot as plt
from matplotlib import cm

# define the scene
normal = np.array([0, 1, 0])  # normal vector of a point
# sample points on a hemisphere
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, np.pi, 100)
theta, phi = np.meshgrid(theta, phi)
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)
view_dir = np.stack((x, y, z), axis=2)


# compute the occluding contour
outline = np.array([np.dot(normal, view_dir[_, __]) <
                    0.1 for _ in range(100) for __ in range(100)])
outline = outline.astype(np.float32)
outline = outline.reshape([100, 100])

# compute lambertian radiance
lambertian = np.ones_like(outline)  # no view-dependent light
lambertian = lambertian.astype(np.float32)
lambertian *= 0.6


# compute phong radiance
light = np.array([0.6, 0.6, 0.1])  # light direction
# compute the specular radiance using the phong model
# the reflect direction
reflect_dir = 2 * np.dot(normal, light) * normal - light

raw = np.array([max(2*np.dot(reflect_dir, view_dir[_, __]), 0)
                for _ in range(100) for __ in range(100)])
raw = raw.reshape([100, 100])
raw = raw+0.1

glossy = np.array([np.power(np.dot(reflect_dir, view_dir[_, __]), 4)
                   for _ in range(100) for __ in range(100)])
glossy = glossy.reshape([100, 100])*2

specular = np.array([np.power(np.dot(reflect_dir, view_dir[_, __]), 10)
                     for _ in range(100) for __ in range(100)])
specular = specular.reshape([100, 100])*5

# compute rimlight (smoothed outline)
rimlight = np.array([np.power(1. - np.dot(normal, view_dir[_, __]), 3)
                     for _ in range(100) for __ in range(100)])
rimlight = rimlight.astype(np.float32)
rimlight = rimlight.reshape([100, 100])

# %%
# visualize the contour on a hemisphere
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta, phi, outline, alpha=0.4, color='b')
# plot the projection of contour to each dimension
ax.contour(theta, phi, outline, zdir='z', offset=-1, cmap=cm.coolwarm)
ax.contour(theta, phi, outline, zdir='x', offset=-1, cmap=cm.coolwarm)
ax.contour(theta, phi, outline, zdir='y', offset=4, cmap=cm.coolwarm)

ax.set(xlim=(-1, 4), ylim=(0, 4), zlim=(-1, 2),
       xlabel='X', ylabel='Y', zlabel='Z')

ax.set_xlabel('theta')
ax.set_ylabel('phi')
ax.set_zlabel('radiance intensity')
plt.title("Radiance function of occluding contour")
plt.show()

# %%visualize the lambertian radiance
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta, phi, lambertian, alpha=0.4, color='b')
# plot the projection of contour to each dimension
ax.contour(theta, phi, lambertian, zdir='z', offset=-1, cmap=cm.coolwarm)
ax.contour(theta, phi, lambertian, zdir='x', offset=-1, cmap=cm.coolwarm)
ax.contour(theta, phi, lambertian, zdir='y', offset=4, cmap=cm.coolwarm)
ax.set(xlim=(-1, 4), ylim=(0, 4), zlim=(-1, 2),
       xlabel='X', ylabel='Y', zlabel='Z')

ax.set_xlabel('theta')
ax.set_ylabel('phi')
ax.set_zlabel('radiance intensity')
plt.title("Radiance function of Lambertian material")
plt.show()

# %% visualize the phong radiance
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta, phi, specular, alpha=0.4, color='b')
# plot the projection of contour to each dimension
ax.contour(theta, phi, specular, zdir='z', offset=-1, cmap=cm.coolwarm)
ax.contour(theta, phi, specular, zdir='x', offset=-1, cmap=cm.coolwarm)
ax.contour(theta, phi, specular, zdir='y', offset=4, cmap=cm.coolwarm)
ax.set(xlim=(-1, 4), ylim=(0, 4), zlim=(-1, 2),
       xlabel='X', ylabel='Y', zlabel='Z')

ax.set_xlabel('theta')
ax.set_ylabel('phi')
ax.set_zlabel('radiance intensity')
plt.title("Radiance function of specular material")
plt.show()

# %% visualize the glossy radiance
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta, phi, glossy, alpha=0.4, color='b')
# plot the projection of contour to each dimension
ax.contour(theta, phi, glossy, zdir='z', offset=-1, cmap=cm.coolwarm)
ax.contour(theta, phi, glossy, zdir='x', offset=-1, cmap=cm.coolwarm)
ax.contour(theta, phi, glossy, zdir='y', offset=4, cmap=cm.coolwarm)
ax.set(xlim=(-1, 4), ylim=(0, 4), zlim=(-1, 2),
       xlabel='X', ylabel='Y', zlabel='Z')

ax.set_xlabel('theta')
ax.set_ylabel('phi')
ax.set_zlabel('radiance intensity')
plt.title("Radiance function for glossy material")
plt.show()

# %% visualize rimlight
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta, phi, rimlight, alpha=0.4, color='b')
# plot the projection of contour to each dimension
ax.contour(theta, phi, rimlight, zdir='z', offset=-1, cmap=cm.coolwarm)
ax.contour(theta, phi, rimlight, zdir='x', offset=-1, cmap=cm.coolwarm)
ax.contour(theta, phi, rimlight, zdir='y', offset=4, cmap=cm.coolwarm)
ax.set(xlim=(-1, 4), ylim=(0, 4), zlim=(-1, 2),
       xlabel='X', ylabel='Y', zlabel='Z')

ax.set_xlabel('theta')
ax.set_ylabel('phi')
ax.set_zlabel('radiance intensity')
plt.title("Radiance function for smoothed outline")
plt.show()

# %%
