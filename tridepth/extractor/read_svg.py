"""
[Reference]
https://github.com/guyc/scadtrace/blob/master/svg.py
"""
import xml.dom.minidom
import re
import string
import numpy as np


class SvgState:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.x0 = 0  # opening of path
        self.y0 = 0
        # Polygon
        self.points = []
        self.edges = []

    def polygon(self):
        return np.array(self.points), np.array(self.edges)


class SvgCmd:
    def __init__(self, match):
        self.cmd = match.group(1)
        self.relative = self.cmd.islower()
        if len(match.groups()) > 1:
            self.setCoords(self.parseCoords(match.group(2)))

    def absolute(self, state, points):
        if self.relative:
            abs = []
            for point in points:
                abs.append([point[0] + state.x, point[1] + state.y])
            return abs
        else:
            return points

    def parseCoords(self, match):
        coords = []
        for coord in re.split('\s+', match.strip()):
            coords.append(float(coord))
        return coords

    def setCoords(self, coords):
        self.coords = coords

    def repack(self, coords, groupSize):
        groups = []
        assert len(coords) % (2 * groupSize) == 0
        while len(coords) > 0:
            group = []
            for _ in range(0, groupSize):
                group.append([coords.pop(0), coords.pop(0)])
            groups.append(group)
        return groups

    def pack(self, coords):
        points = []
        assert len(coords) % 2 == 0
        while len(coords) > 0:
            points.append([coords.pop(0), coords.pop(0)])
        return points


class SvgMoveCmd(SvgCmd):
    def setCoords(self, coords):
        assert len(coords) == 2
        self.coords = coords

    def run(self, state):
        coords = self.absolute(state, [self.coords])
        # Update states
        state.x = coords[0][0]
        state.y = coords[0][1]
        # Starts a new subpath
        state.x0 = state.x
        state.y0 = state.y
        # Add points
        state.points.append([state.x, state.y])
        return state


class SvgCurveCmd(SvgCmd):
    def setCoords(self, coords):
        assert (len(coords) % 6) == 0
        self.curves = self.repack(coords, 3)  # each curve has 3 x,y coords

    def divide_points(self, state, curve, divisions=10):
        """
        [ref] http://www.c-sharpcorner.com/uploadfile/apundit/drawingcurves11182005012515am/drawingcurves.aspx
        """
        curve = self.absolute(state, curve)

        p1x = state.x
        p1y = state.y
        p2x = curve[0][0]
        p2y = curve[0][1]
        p3x = curve[1][0]
        p3y = curve[1][1]
        p4x = curve[2][0]
        p4y = curve[2][1]

        # Start point
        state.points.append([p1x, p1y])

        # http://en.wikipedia.org/wiki/B%C3%A9zier_curve
        for i in range(1, divisions):
            t = float(i) / float(divisions)
            f1 = (1 - t)**3
            f2 = 3 * (1 - t)**2 * t
            f3 = 3 * (1 - t) * t * t
            f4 = t * t * t
            # Divided points
            div_px = f1 * p1x + f2 * p2x + f3 * p3x + f4 * p4x
            div_py = f1 * p1y + f2 * p2y + f3 * p3y + f4 * p4y

            # Update state
            state.points.append([div_px, div_py])
            last_id = len(state.points) - 1
            state.edges.append([last_id - 1, last_id])

        # Final point
        state.points.append([p1x, p1y])
        last_id = len(state.points) - 1
        state.edges.append([last_id - 1, last_id])

        # Update x,y in state
        state.x = p4x
        state.y = p4y

        return state

    def run(self, state):
        for curve in self.curves:
            state = self.divide_points(state, curve)
        return state


class SvgCloseCmd(SvgCmd):
    def run(self, state):
        state.x = state.x0
        state.y = state.y0
        # Update
        state.points.append([state.x0, state.y0])
        last_id = len(state.points) - 1
        state.edges.append([last_id - 1, last_id])
        return state


class SvgLineCmd(SvgCmd):
    def setCoords(self, coords):
        assert(len(coords) % 2) == 0  # expect x,y pairs
        self.strokes = self.pack(coords)

    def run(self, state):
        for stroke in self.strokes:
            absStroke = self.absolute(state, [stroke])
            point = absStroke[0]
            # Update state
            state.x = point[0]
            state.y = point[1]
            # Update polygon
            state.points.append([state.x, state.y])
            last_id = len(state.points) - 1
            state.edges.append([last_id - 1, last_id])
        return state


class SVGReader:
    def __init__(self, svg_string):
        self.doc = xml.dom.minidom.parseString(svg_string)
        self.re = {
            'M': re.compile('^\s*(M)\s*((-?[0-9\.]+\s*){2})'),
            'C': re.compile('^\s*(c|C)\s*((-?[0-9\.]+\s*){6,})\s*'),
            'Z': re.compile('^\s*(z)\s*'),
            'L': re.compile('^\s*(L)\s*((-?[0-9\.]+\s*){2,})\s*'),
        }
        svg_node = self.doc.documentElement
        self.width = float(svg_node.attributes["width"].value.rstrip("pt"))
        self.height = float(svg_node.attributes["height"].value.rstrip("pt"))

        # Prepare for polygon extraction
        self.paths = self._extract_paths()
        self.state = SvgState()

    def _parse_path(self, path_string):
        cmds = []
        while path_string != "":
            match = self.re['M'].match(path_string)
            if match:
                cmds.append(SvgMoveCmd(match))
            else:
                match = self.re['C'].match(path_string)
                if match:
                    cmds.append(SvgCurveCmd(match))
                else:
                    match = self.re['Z'].match(path_string)
                    if match:
                        cmds.append(SvgCloseCmd(match))
                    else:
                        match = self.re['L'].match(path_string)
                        if match:
                            cmds.append(SvgLineCmd(match))
                        else:
                            raise Exception('Unparseable path string (' + path_string + ')')

            # trim whatever was matched
            path_string = match.string[match.end(0):]
        return cmds

    def _extract_paths(self):
        paths = []
        for node in self.doc.getElementsByTagName('path'):
            d = node.attributes["d"].value
            paths.append(self._parse_path(d))
        return paths

    def run(self):
        for path in self.paths:
            for cmd in path:
                self.state = cmd.run(self.state)
        return self.state.polygon()
