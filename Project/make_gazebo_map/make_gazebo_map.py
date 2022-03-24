"""This module creates gazebo-map by subscribing to the occupancy grid map."""

import rospy as ros
import numpy as np
import trimesh
import cv2 as cv
from nav_msgs.msg import OccupancyGrid


class MapConverter (object):
    """Class to subscribe and develop gazebo map."""

    #
    def __init__(self, map_topic, threshold=1, height=2.0):
        """Initilize the class."""
        self.map_pub = ros.Publisher("occ_map", OccupancyGrid, latch=True,
                                     queue_size=1)

        ros.Subscriber(map_topic, OccupancyGrid, self.map_callback)
        self.threshold = threshold
        self.height = height

    #
    def map_callback(self, map_msg):
        """Create callback for subscibtion."""
        ros.loginfo("Map has been read.")
        map_dims = (map_msg.info.height, map_msg.info.width)
        map_array = np.array(map_msg.data).reshape(map_dims)

        # setting all negatives (unknowns) to zero (unoccupied)
        map_array[map_array < 0] = 0

        contours = self.get_occupied_regions(map_array)
        meshes = [self.contour_to_mesh(c, map_msg.info) for c in contours]

        corners = list(np.vstack(contours))
        corners = [c[0] for c in corners]
        self.publish_test_map(corners, map_msg.info, map_msg.header)
        mesh = trimesh.util.concatenate(meshes)

        # exporting mesh
        mesh_type = ros.get_param("~mesh_type", "stl")
        export_dir = ros.get_param("~export_dir")
        if mesh_type == "stl":
            with open(export_dir+"/map.stl", "w") as f:
                mesh.export(f, "stl")
            ros.loginfo("Exported stl map, you can shutdown the node")
        elif mesh_type == "dae":
            with open(export_dir+"/map.dae", "w") as f:
                f.write(trimesh.exchange.dae.export_collada(mesh))
            ros.loginfo("Exported dae map, you can shutdown the node")

    #
    def publish_test_map(self, points, metadata, map_header):
        """
        For testing purposes, it publishes a map highlighting certain points.

        Points (points) is a list of tuples (x, y) in the map's
        coordinate system.
        """
        test_map = np.zeros((metadata.height, metadata.width))
        for x, y in points:
            test_map[y, x] = 100
        test_map_msg = OccupancyGrid()
        test_map_msg.header = map_header
        test_map_msg.header.stamp = ros.Time.now()
        test_map_msg.info = metadata
        test_map_msg.data = list(np.ravel(test_map))
        self.map_pub.publish(test_map_msg)

    #
    def get_occupied_region(self, map_array):
        """Get occupied regions from map."""
        map_array = np.uint8(map_array)
        map_array = cv.threshold(map_array, self.threshold, 100,
                                 cv.THRESH_BINARY)[1]

        contours, hierarchy = cv.findContours(map_array, cv.RETR_CCOMP,
                                              cv.CHAIN_APPROX_NONE)

        hierarchy = hierarchy[0]
        # corner_idxs = [i for i in range(len(contours))
        #                if hierarchy[i][3] == -1]
        # # return [contours[i] for i in corner_idxs]
        return contours

    #
    def contour_to_mesh(self, contour, metadata):
        """Convert contours to mesh."""
        height = np.array([0, 0, self.height])
        meshes = []
        for point in contour:
            x, y = point[0][0]
            vertices = []
            new_vertices = [
                coords_to_loc((x, y), metadata),
                coords_to_loc((x, y+1), metadata),
                coords_to_loc((x+1, y), metadata),
                coords_to_loc((x+1, y+1), metadata)]
            vertices.extend(new_vertices)
            vertices.extend([v+height for v in new_vertices])
            faces = [[0, 2, 4],
                     [4, 2, 6],
                     [1, 2, 0],
                     [3, 2, 1],
                     [5, 0, 4],
                     [1, 0, 5],
                     [3, 7, 2],
                     [7, 6, 2],
                     [7, 4, 6],
                     [5, 4, 7],
                     [1, 5, 3],
                     [7, 3, 5]]
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            if not mesh.is_volume():
                ros.logdebug("Fixing mesh normals")
                mesh.fix_normals()
            meshes.append(mesh)

        mesh = trimesh.util.concatenate(meshes)
        mesh.remove_duplicate_faces()
        # mesh will still have internal faces.  Would be better to get
        # all duplicate faces and remove both of them, since duplicate faces
        # are guaranteed to be internal faces
        return mesh


def coords_to_loc(coords, metadata):
    """Convert coords to location."""
    x, y = coords
    loc_x = x*metadata.resolution+metadata.origin.position.x
    loc_y = y*metadata.resolution+metadata.origin.position.y

    # TODO: transform (x*res, y*res, 0.0) by Pose map_metadata.origin
    # instead of assuming origin is at z=0 with no rotation wrt map frame

    return np.array([loc_x, loc_y, 0])
