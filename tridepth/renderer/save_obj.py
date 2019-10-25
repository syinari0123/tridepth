# MIT License
#
# Copyright (c) 2018 Yao Feng
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
from skimage.io import imsave


def write_obj_with_texture(obj_name, verts, faces, textures, uv_coords):
    """
    Save 3D face model with texture represented by texture map.

    Args:
        obj_name (str):
        verts (ndarray): [N,3]
        faces (ndarray): [M,3], composed of 3 vert-ids for each face.
        textures (ndarray): scene image ([228,304,3])
        uv_coords (ndarray): [N,2](range=[0,1]) which determine the mapping to textures.
    """
    # Prepare filename
    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'
    mtl_name = obj_name.replace('.obj', '.mtl')
    texture_name = obj_name.replace('.obj', '_texture.png')

    faces = faces.copy()
    faces += 1  # mesh lab start with 1

    # Write into obj file
    with open(obj_name, 'w') as f:
        # First line: write mtlib (material library)
        s = "mtllib {}\n".format(os.path.basename(mtl_name))
        f.write(s)

        # Write verts
        for i in range(verts.shape[0]):
            s = 'v {} {} {}\n'.format(verts[i, 0], verts[i, 1], verts[i, 2])
            f.write(s)

        # Write uv coords
        for i in range(uv_coords.shape[0]):
            s = 'vt {} {}\n'.format(uv_coords[i, 0], 1 - uv_coords[i, 1])
            f.write(s)

        f.write("usemtl SceneTexture\n")

        # write f: ver ind/ uv ind
        for i in range(faces.shape[0]):
            s = 'f {}/{} {}/{} {}/{}\n'.format(
                faces[i, 2], faces[i, 2], faces[i, 1], faces[i, 1], faces[i, 0], faces[i, 0])
            f.write(s)

    # Write mtl
    with open(mtl_name, 'w') as f:
        f.write("newmtl SceneTexture\n")
        s = 'map_Kd {}\n'.format(os.path.basename(texture_name))  # map to image
        f.write(s)

    # Write texture as png
    imsave(texture_name, textures)
