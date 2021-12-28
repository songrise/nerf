import json
import numpy as np
import open3d

res_1 = []
res_2 = []
with open("transforms_train_1.json") as fd:
    meta = json.load(fd)
    frames = meta["frames"]
    # extract transform matrix
    for frame in frames:
        mat = frame["transform_matrix"]
        mat = np.array(mat)
        # extract the translation part
        translation = np.array(mat[:3, 3])

        res_1.append(translation)

    # add origin for reference
    res_1.append(np.array([0, 0, 0]))


with open("transforms_train_2.json") as fd:
    meta = json.load(fd)
    frames = meta["frames"]
    # extract transform matrix
    for frame in frames:
        mat = frame["transform_matrix"]
        mat = np.array(mat)
        # extract the translation part
        translation = np.array(mat[:3, 3])
        res_2.append(translation)

    # add origin for reference
    res_2.append(np.array([0, 0, 0]))


# visualize point cloud
pcd_1 = open3d.geometry.PointCloud()
pcd_1.points = open3d.utility.Vector3dVector(res_1)
pcd_1.paint_uniform_color(np.array([1, 0, 0]))  # red

pcd_2 = open3d.geometry.PointCloud()
pcd_2.points = open3d.utility.Vector3dVector(res_2)
pcd_2.paint_uniform_color(np.array([0, 1, 0]))  # green

mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.6, origin=[0, 0, 0])

# Red: our data, Green: offical data
print("radius are {} {}".format(np.linalg.norm(
    res_1[0]), np.linalg.norm(res_2[0])))
open3d.visualization.draw_geometries([pcd_1, pcd_2, mesh_frame])
