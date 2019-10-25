from __future__ import division
import os
import torch
import numpy as np
import scipy.spatial


def depth2point3d(depth, intrinsics, output_size):
    """
    """
    # extract intrinsic params
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    xx, yy = np.meshgrid(np.arange(0, output_size[1]), np.arange(0, output_size[0]))
    x = (xx - cx) / fx
    y = (yy - cy) / fy
    pos3d = np.dstack((x * depth, y * depth, depth)).reshape(-1, 3)
    pixels_tri = np.dstack((x, y)).reshape(-1, 2)  # [H*W, 2]
    return pos3d, pixels_tri


def write_obj(obj_name, vertices, colors, triangles):
    """Save 3D face model

    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        colors: shape = (nver, 3)
        triangles: shape = (ntri, 3)
    """
    print("Saving {}...".format(obj_name))

    triangles = triangles.copy()
    triangles += 1  # meshlab start with 1

    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'

    # write obj
    with open(obj_name, 'w') as f:

        # write vertices & colors
        for i in range(vertices.shape[0]):
            # s = 'v {} {} {} \n'.format(vertices[0,i], vertices[1,i], vertices[2,i])
            s = 'v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1],
                                               vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2])
            f.write(s)

        # write f: ver ind/ uv ind
        [k, ntri] = triangles.shape
        for i in range(triangles.shape[0]):
            # s = 'f {} {} {}\n'.format(triangles[i, 0], triangles[i, 1], triangles[i, 2])
            s = 'f {} {} {}\n'.format(triangles[i, 2], triangles[i, 1], triangles[i, 0])
            f.write(s)

    print("complate!")


def load_obj(filename_obj, device=torch.device("cuda")):
    """
    Load Wavefront .obj file.
    This function only supports vertices (v x x x) and faces (f x x x).
    """
    with open(filename_obj) as f:
        lines = f.readlines()

    vertices = []
    faces = []
    for i, line in enumerate(lines):
        if len(line.split()) == 0:
            continue
        # vertices
        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])
        # faces
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split('/')[0])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split('/')[0])
                v2 = int(vs[i + 2].split('/')[0])
                faces.append((v0, v1, v2))

    # Integrate (TODO: too slow???)
    vertices = torch.from_numpy(np.vstack(vertices).astype(np.float32)).to(device)
    faces = torch.from_numpy(np.vstack(faces).astype(np.int32)).to(device) - 1

    # Normalize into a unit cube centered zero
    # if normalization:
    #    vertices -= vertices.min(0)[0][None, :]
    #    vertices /= torch.abs(vertices).max()
    #    vertices *= 2
    #    vertices -= vertices.max(0)[0][None, :] / 2

    return vertices.unsqueeze(0), faces.unsqueeze(0)


def create_grid_mesh(np_img, np_depth, np_intrinsics, obj_name):
    """
    """
    height, width = np_depth.shape
    # vertices / uv_map
    vertices, pixels = depth2point3d(np_depth, np_intrinsics, output_size=(height, width))

    # faces
    tri = scipy.spatial.Delaunay(pixels)
    faces = np.array(tri.simplices.tolist())  # index
    colors = np_img.reshape(-1, 3)

    # Save format
    write_obj(obj_name, vertices, colors, faces)

    return vertices.shape[0], faces.shape[0]
