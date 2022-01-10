import json
import numpy as np
import open3d

trans_1 = []
rot_1 = []
trans_2 = []
rot_2 = []
with open("transforms_train_1.json") as fd:
    meta = json.load(fd)
    frames = meta["frames"]

    # extract transform matrix
    for frame in frames:
        mat = frame["transform_matrix"]
        mat = np.array(mat)
        # extract the translation part
        translation = np.array(mat[:3, 3])
        # extract the rotation part
        rotation = np.array(mat[:3, :3])
        trans_1.append(translation)
        rot_1.append(rotation)

    # add origin for reference
    trans_1.append(np.array([0, 0, 0]))


with open("transforms_train_2.json") as fd:
    meta = json.load(fd)
    frames = meta["frames"]
    # extract transform matrix
    for frame in frames:
        mat = frame["transform_matrix"]
        mat = np.array(mat)
        # extract the translation part
        translation = np.array(mat[:3, 3])
        # extract the rotation part
        rotation = np.array(mat[:3, :3])

        trans_2.append(translation)
        rot_2.append(rotation)

    # add origin for reference
    trans_2.append(np.array([0, 0, 0]))

# visualize the gaze vector for each dataset


# visualize point cloud
pcd_1 = open3d.geometry.PointCloud()
pcd_1.points = open3d.utility.Vector3dVector(trans_1)
pcd_1.paint_uniform_color(np.array([1, 0, 0]))  # red

pcd_2 = open3d.geometry.PointCloud()
pcd_2.points = open3d.utility.Vector3dVector(trans_2)
pcd_2.paint_uniform_color(np.array([0, 1, 0]))  # green

# world coordinate
mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.6, origin=[0, 0, 0])

# local coordinate for 1st point in cloud 1
mesh_frame_1 = open3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.6, origin=trans_1[0]).rotate(rot_1[0])

# local coordinate for 1st point in cloud 2
mesh_frame_2 = open3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.6, origin=trans_2[0]).rotate(rot_2[0])

# Red: our data, Green: offical data
print("radius are {} {}".format(np.linalg.norm(
    trans_1[0]), np.linalg.norm(trans_2[0])))


open3d.visualization.draw_geometries(
    [pcd_1, pcd_2, mesh_frame, mesh_frame_1, mesh_frame_2])
