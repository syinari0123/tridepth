import pymesh
import numpy as np
from numpy.linalg import norm


def add_frame(wires, output_size):
    height, width = output_size

    # Prepare frame
    frame_vertices = np.array([
        [0.0, 0.0],
        [0.0, height],
        [width, height],
        [width, 0.0]
    ])
    frame_edges = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
    ])

    if wires.num_vertices == 0:
        wires.load(frame_vertices, frame_edges)
        return wires
    else:
        vertices = wires.vertices
        edges = wires.edges

        frame_edges = frame_edges + wires.num_vertices

        vertices = np.vstack([vertices, frame_vertices])
        edges = np.vstack([edges, frame_edges])
        wires.load(vertices, edges)
        return wires


def resolve_self_intersection(wires, min_edge_size=1):
    vertices, edges = pymesh.snap_rounding(wires.vertices, wires.edges, 1.0 * min_edge_size)  # TODO
    wires.load(vertices, edges)
    return wires


def cleanup(wires):
    # Remove duplicated edges.
    ordered_edges = np.sort(wires.edges, axis=1)
    __, unique_edge_ids, __ = pymesh.unique_rows(ordered_edges)
    edges = wires.edges[unique_edge_ids, :]
    wires.load(wires.vertices, edges)

    # Remove topologically degenerate edges.
    is_not_topologically_degenerate = edges[:, 0] != edges[:, 1]
    if not np.all(is_not_topologically_degenerate):
        wires.filter_edges(is_not_topologically_degenerate)

    return wires
