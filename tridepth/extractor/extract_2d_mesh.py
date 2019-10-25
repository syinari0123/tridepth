import os
import sys
import tempfile
import subprocess
import cv2
import pymesh
import numpy as np
import torch
import triangle as tr

from tridepth import BaseMesh
from tridepth.extractor import calculate_canny_edges
from tridepth.extractor import SVGReader
from tridepth.extractor import resolve_self_intersection, cleanup
from tridepth.extractor import add_frame


class Mesh2DExtractor:
    def __init__(self, canny_params={"denoise": False}, at_params={"filter_itr": 4, "error_thresh": 0.01}):
        self.canny_params = canny_params  # TODO
        self.autotrace_cmd = ['autotrace',
                              '--centerline',
                              '--remove-adjacent-corners',
                              '--filter-iterations', str(at_params["filter_itr"]),
                              '--error-threshold', str(at_params["error_thresh"]),
                              '--input-format=bmp',
                              '--output-format=svg']

    def _execute_autotrace(self, filename, debug=False):
        """Execute autotrace with input (bmp-file)
            - https://github.com/autotrace/autotrace
        Returns:
            svg_string: string starting from '<svg/>'
        """
        # Execute autotrace
        p = subprocess.Popen(self.autotrace_cmd + [filename], stdout=subprocess.PIPE)

        # Read the converted svg contents
        svg_string = p.communicate()[0]
        if not len(svg_string):
            print("autotrace_cmd: " + ' '.join(self.autotrace_cmd + [filename]), file=sys.stderr)
            print("ERROR: returned nothing, leaving tmp bmp file around for you to debug", file=sys.stderr)
            sys.exit(1)
        else:
            if debug:
                print(filename)
                sys.exit(1)
            else:
                os.unlink(filename)  # Remove the tempolary file

        return svg_string

    def _read_polygon_from_svg(self, svg_string):
        """
        """
        # Extract polygon information from svg-string
        #   - https://github.com/guyc/scadtrace/blob/master/svg.py
        svg_reader = SVGReader(svg_string)
        verts_2d, edges = svg_reader.run()

        # Store polygons as wire-format (w/ cleaning)
        #   - https://github.com/PyMesh/PyMesh/blob/master/scripts/svg_to_mesh.py
        if verts_2d.shape[0] == 0:
            wires = pymesh.wires.WireNetwork.create_empty()
        else:
            wires = pymesh.wires.WireNetwork.create_from_data(verts_2d, edges)
            wires = resolve_self_intersection(wires, min_edge_size=1.5)
            wires = cleanup(wires)

        return wires

    def _triangulation(self, np_edge, wires, output_size, debug=False):
        """
        """
        height, width = output_size

        # We use cython wrapper of Triangle,
        # since other implementations (Pymesh) can't output edges :(
        #   - https://github.com/drufat/triangle
        input_dic = {}
        input_dic["vertices"] = wires.vertices.copy()
        input_dic["segments"] = wires.edges.copy()
        # [Options]
        #   p: Triangulates a Planar Straight Line Graph.
        #   q: no angles smaller than 20 degrees
        try:
            t = tr.triangulate(input_dic, 'pq')
        except:
            import uuid
            unique_filename = str(uuid.uuid4()) + ".png"
            print(wires.vertices.shape, wires.edges.shape)
            cv2.imwrite(unique_filename, np_edge)
            exit()

        if debug:
            import matplotlib.pyplot as plt
            plt.gca().invert_yaxis()
            # plt.imshow(np_edge)
            for edge in wires.edges:
                v1x, v1y = wires.vertices[edge[0]]
                v2x, v2y = wires.vertices[edge[1]]
                plt.plot([v1x, v2x], [v1y, v2y], 'k-', color='r', linewidth=1.0)

            for tri in t['triangles']:
                v1x, v1y = t['vertices'][tri[0]]
                v2x, v2y = t['vertices'][tri[1]]
                v3x, v3y = t['vertices'][tri[2]]
                plt.plot([v1x, v2x], [v1y, v2y], 'k-', color='black', linewidth=1.0)
                plt.plot([v2x, v3x], [v2y, v3y], 'k-', color='black', linewidth=1.0)
                plt.plot([v3x, v1x], [v3y, v1y], 'k-', color='black', linewidth=1.0)

            plt.scatter(wires.vertices[:, 0], wires.vertices[:, 1], s=3.0, c="black")
            plt.show()
            print(t['vertices'].shape, t['triangles'].shape)
            exit()

        # Normalize (range=[0,1])
        vertices = t["vertices"]
        t["vertices"] = np.concatenate((vertices[:, :1] / width,
                                        vertices[:, 1:2] / height,
                                        vertices[:, 2:]), 1)
        t["edgemap"] = np_edge

        return t

    def __call__(self, np_scene):
        """
        Args:
            np_scene: [H,W,3] (ndarray, uint8)
        """
        height, width, _ = np_scene.shape

        # Calculate canny edge
        np_edge, _ = calculate_canny_edges(np_scene, denoise=self.canny_params["denoise"])

        # Save into temp file as bmp-format
        with tempfile.NamedTemporaryFile(suffix='.bmp', delete=False) as temp:
            cv2.imwrite(temp.name, np_edge)

        # Execute vectorization (by Autotrace)
        svg_string = self._execute_autotrace(temp.name)

        # Extract polygon information
        wires = self._read_polygon_from_svg(svg_string)

        # Triangulation
        wires = add_frame(wires, output_size=(height, width))
        mesh_dic = self._triangulation(np_edge, wires, output_size=(height, width))

        # Finally integrate all the information, and create disconnected mesh
        mesh = BaseMesh(mesh_dic)

        return mesh
