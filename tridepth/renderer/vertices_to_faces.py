# MIT License
#
# Copyright (c) 2017 Hiroharu Kato
# Copyright (c) 2018 Nikos Kolotouros
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

import torch


def vertices_to_faces(verts, faces):
    """
    :param verts: [batch size, number of verts, 2 or 3]
    :param faces: [batch size, number of faces, 3)
    :return: [batch size, number of faces, 3, 2or3]
    """
    assert (faces.dim() == 3)
    assert (verts.shape[0] == faces.shape[0])
    assert (verts.shape[2] == 2 or verts.shape[2] == 3)
    assert (faces.shape[2] == 3)

    verts_dim = verts.shape[2]  # 2 or 3

    bs, nv = verts.shape[:2]
    bs, nf = faces.shape[:2]
    device = verts.device
    faces = faces + \
        (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    verts = verts.reshape((bs * nv, verts_dim))
    # pytorch only supports long and byte tensors for indexing
    return verts[faces.long()]
