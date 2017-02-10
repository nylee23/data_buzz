#! /usr/bin/env python
#
# Program: visaulize_dstl
#
# Author: Nick Lee
#
# Description: Preliminary code to visualize and play around with data from DSTL kaggle competition
#
# Date: Feb 3, 2017
#

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tiff
# from shapely.geometry import Point
from shapely.wkt import loads
import shapely
# from descartes import PolygonPatch
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from PIL import Image, ImageDraw


class Train_DSTL(object):
    """
    Class to hold methods for training detection of objects in DSTL images
    """
    def __init__(self):
        # Save useful tables
        self.training_set = pd.read_csv('train_wkt_v4.csv')
        self.grid_sizes = pd.read_csv('grid_sizes.csv')
        self.grid_sizes.columns = pd.Index(['ImageID', 'Xmax', 'Ymin'])
        self.object_colors = {1: 'gray', 2: 'blue', 3: 'black', 4: 'brown', 5: 'green', 6: 'yellow', 7: 'turquoise', 8: 'blue', 9: 'red', 10: 'orange'}

    def display_three_band(self, image_id='6120_2_2'):
        """ Display a given map """
        # fig, ax = plt.subplots()
        img_filename = 'three_band/{:}.tif'.format(image_id)
        image = tiff.imread(img_filename).transpose([1, 2, 0])
        # ax.imshow(255 * self._scale_percentile(image), cmap='cubehelix')
        fig, ax, _ = tiff.imshow(255 * self._scale_percentile(image))
        masks = self._get_objects(image_id, ax=ax)
        plt.show()
        return image, masks

    def _get_objects(self, image_id, ax=None):
        """ Find all the objects defined in the training set """
        if ax is None:
            fig, ax = plt.subplots()
        # Find xy_limits
        img_size, xy_limits = self._get_image_properties(image_id)
        # Loop over all classes that aren't empty
        objects = self.training_set.loc[(self.training_set['ImageId'] == image_id) & (self.training_set['MultipolygonWKT'] != 'MULTIPOLYGON EMPTY')]
        masks = []
        for idx, row in objects.iterrows():
            masks.append(self._make_mask(row['MultipolygonWKT'], img_size, xy_limits, class_type=row['ClassType'], ax=ax))
        return masks

    def _make_mask(self, multi_polygon_wkt, img_size, xy_limits, class_type=1, ax=None):
        """
        Take a multi-polygon and make an image mask that selects all pixels that lie within the multi-polygon
        """
        # Translate wkt to Shapely Polygon
        shapes = loads(multi_polygon_wkt)  # WKT to Shapely format
        # Make a True/False image using PIL
        img = Image.new('1', img_size[-1::-1], 0)
        # Fill in the Image with the polygons
        for shape in shapes:
            xy_perimeter = np.array(shape.exterior.coords)
            xy_scaled = dstl._convert_xy_to_img_scale(xy_perimeter, img_size, xy_limits)
            poly = [(coord[0], coord[1]) for coord in xy_scaled]
            ImageDraw.Draw(img).polygon(poly, fill=1)
        mask = np.array(img)
        # Plot mask
        if ax is not None:
            ax.contour(mask, colors=self.object_colors[class_type])
        return mask

    # Helper Methods
    def _get_image_properties(self, image_id='6120_2_2'):
        """ Return the img_size and xy_limits of a given image """
        img_filename = 'three_band/{:}.tif'.format(image_id)
        image = tiff.imread(img_filename).transpose([1, 2, 0])
        img_size = image.shape[0:2]
        xy_limits = self.grid_sizes.loc[self.grid_sizes['ImageID'] == image_id, ['Xmax', 'Ymin']].values.reshape(-1)
        return img_size, xy_limits

    def _convert_xy_to_img_scale(self, xy, img_size, xy_limits):
        """
        Convert a list of xy tuples that are between [0, 1] and [-1, 0] and convert to the corresponding coordinates of an image

        Follows transformation provided here:
        https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/details/data-processing-tutorial

        Args:
            xy - list of (x, y) tuples where x & y are normalized
            img_size - Tuple of image (height, width)
            xy_limits - Tuple of Xmax and Ymin needed to properly scale image
        Returns:
            x_scaled, y_scaled - x and y coordinates scaled to the image
        """
        # Extract arrays of x and y
        x, y = np.array(xy).T
        h, w = img_size
        xmax, ymin = xy_limits
        x_scaled = (w * w / (w + 1)) * x / xmax
        y_scaled = (h * h / (h + 1)) * y / ymin
        xy_scaled = np.vstack([x_scaled, y_scaled]).T
        return xy_scaled

    def _convert_img_to_xy_scale(self, xy, img_size, xy_limits):
        """ Convert from coordinates on the image scale back to the normalized xy scale of the Polygons

        Args:
            xy - list of (x, y) tuples where x & y are in image coordinates
            img_size - Tuple of image (height, width)
            xy_limits - Tuple of Xmax and Ymin needed to properly scale image
        Returns:
            x_norm, y_norm - x and y coordinates normalized to be between 0 < x < 1 and -1 < y < 0
        """
        x, y = np.array(xy).T
        h, w = img_size
        xmax, ymin = xy_limits
        x_norm = x * xmax / (w * w / (w + 1))
        y_norm = y * ymin / (h * h / (h + 1))
        xy_norm = np.vstack([x_norm, y_norm]).T
        return xy_norm

    def _scale_percentile(self, image):
        """
        Fixes the pixel value range to 2%-98% original distribution of values

        As seen at:
        https://www.kaggle.com/chatcat/dstl-satellite-imagery-feature-detection/load-a-3-band-tif-image-and-overlay-dirt-tracks
        """
        orig_shape = image.shape
        image = np.reshape(image, [image.shape[0] * image.shape[1], 3]).astype(float)
        # Get 2nd and 98th percentile
        mins = np.percentile(image, 1, axis=0)
        maxs = np.percentile(image, 99, axis=0) - mins
        image = (image - mins[None, :]) / maxs[None, :]
        image = np.reshape(image, orig_shape)
        image = image.clip(0, 1)
        return image

    # Old code - can probably be removed
    def plot_polygons(self, multi_polygon_wkt, img_size, xy_limits, class_type=1, ax=None):
        """ Note: Not necessary anymore. """
        # Translate wkt to Shapely Polygon
        shapes = loads(multi_polygon_wkt)  # WKT to Shapely format
        all_patches = []
        for shape in shapes:
            xy_perimeter = np.array(shape.exterior.coords)  # perimeter of polygon
            xy_scaled = self._convert_xy_to_img_scale(xy_perimeter, img_size, xy_limits)
            all_patches.append(mpatches.Polygon(xy_scaled, fill=False, color=self.object_colors[class_type], zorder=2))
        patches = PatchCollection(all_patches, match_original=True)
        ax.add_collection(patches)


# Testing functions used to figure out and understand code #
def plot_object_test():
    # Get image
    dstl = Train_DSTL()
    image_id = dstl.training_set.loc[3, 'ImageId']
    img_filename = 'three_band/{:}.tif'.format(image_id)
    image = tiff.imread(img_filename).transpose([1, 2, 0])
    xy_limits = dstl.grid_sizes.loc[dstl.grid_sizes['ImageID'] == image_id, ['Xmax', 'Ymin']].values.reshape(-1)
    # Load polygons
    multi_polygon_wkt = dstl.training_set.loc[3, 'MultipolygonWKT']
    shapes = loads(multi_polygon_wkt)
    shape = shapes[0]
    xy_perimeter = np.array(shape.exterior.coords)  # perimeter of polygon
    xy_scaled = dstl._convert_xy_to_img_scale(xy_perimeter, image.shape[0:2], xy_limits)
    # Plot
    fig, ax = plt.subplots()
    patch = mpatches.Polygon(xy_scaled)
    ax.add_patch(patch)
    ax.set_xlim(np.min(xy_scaled.T[0]) * 0.95, np.max(xy_scaled.T[0]) * 1.05)
    ax.set_ylim(np.min(xy_scaled.T[1]) * 0.95, np.max(xy_scaled.T[1]) * 1.05)
    plt.show()


def find_points_inside_test(all_patches=True):
    # Get a polygon
    dstl = Train_DSTL()
    image_id = '6120_2_2'
    img_size, xy_limits = dstl._get_image_properties(image_id)
    shapes = loads(dstl.training_set.loc[11, 'MultipolygonWKT'])
    if all_patches:
        xv, yv = np.meshgrid(range(img_size[1]), range(img_size[0]))
        xy_grid = np.vstack((xv.reshape(-1), yv.reshape(-1))).T
        mask = [shapes.contains(shapely.geometry.Point(xy)) for xy in xy_grid]

        # patch_list = []
        # for shape in shapes:
        #     xy_perimeter = np.array(shape.exterior.coords)
        #     xy_scaled = dstl._convert_xy_to_img_scale(xy_perimeter, img_size, xy_limits)
        #     patch_list.append(mpatches.Polygon(xy_scaled, fill=False, color='red', zorder=2))
        # patches = PatchCollection(patch_list, match_original=True)
        # paths = patches.get_paths()
        return mask
        # mask = [path.contains_points(xy_grid) for path in paths]
        # return mask
    else:
        xy_perimeter = np.array(shapes[np.random.choice(range(len(shapes)))].exterior.coords)
        xlim = (np.min(xy_perimeter.T[0]) * 0.99, np.max(xy_perimeter.T[0]) * 1.01)
        ylim = (np.min(xy_perimeter.T[1]) * 0.99, np.max(xy_perimeter.T[1]) * 1.01)
        patch = mpatches.Polygon(xy_perimeter, zorder=1)
        # Find points inside polygon
        path = patch.get_path()
        xv, yv = np.meshgrid(np.linspace(xlim[0], xlim[1], 10), np.linspace(ylim[0], ylim[1], 10))
        xy_grid = np.vstack((xv.reshape(-1), yv.reshape(-1))).T
        mask = path.contains_points(xy_grid)
        # Plot
        fig, ax = plt.subplots()
        ax.add_patch(patch)
        ax.plot(*xy_grid[mask].T, 'ro', zorder=2)
        ax.plot(*xy_grid[~mask].T, 'ko', zorder=2)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        plt.show()


def make_mask(method='pandas'):
    """
    Test different ways to select all pixels within a given MultiPolygon
    """
    dstl = Train_DSTL()
    image_id = '6120_2_2'
    img_size, xy_limits = dstl._get_image_properties(image_id)
    shapes = loads(dstl.training_set.loc[11, 'MultipolygonWKT'])
    xv, yv = np.meshgrid(range(img_size[1]), range(img_size[0]))
    xy_grid = np.vstack((xv.reshape(-1), yv.reshape(-1))).T  # All possible pixels

    if method == 'pandas':
        pass
    elif method == 'bounds':
        for shape in shapes:
            xy_perimeter = np.array(shape.exterior.coords)
            xy_scaled = dstl._convert_xy_to_img_scale(xy_perimeter, img_size, xy_limits)
            # Keep track of boundaries
            bounds = shape.bounds
            xy_lim = dstl._convert_xy_to_img_scale([bounds[:2], bounds[2:]], img_size, xy_limits)
            xy_min = xy_lim[0].round()
            xy_max = xy_lim[1].round()
            x_vec = np.arange(xy_min[0] - 1, xy_min[0] + 2)
            y_vec = np.arange(xy_min[1] - 1, xy_min[1] + 2)
            points_in_poly = np.array(np.meshgrid(x_vec, y_vec)).T.reshape(-1, 2)
            try:
                possible_pixels = np.vstack((possible_pixels, points_in_poly))
            except:
                possible_pixels = points_in_poly
        norm_coords = dstl._convert_img_to_xy_scale(possible_pixels, img_size, xy_limits)
        # %timeit shapes.contains(shapely.geometry.Point(norm_coords[0]))
        # 1 loop, best of 3: 495 ms per loop
        return norm_coords
    elif method == 'patches' or method == 'pil':
        patch_list = []
        img = Image.new('1', img_size, 0)
        for shape in shapes:
            xy_perimeter = np.array(shape.exterior.coords)
            xy_scaled = dstl._convert_xy_to_img_scale(xy_perimeter, img_size, xy_limits)
            patch = mpatches.Polygon(xy_scaled, fill=False, color='red', zorder=2)
            patch_list.append(patch)
            # PIL
            path = patch.get_path()
            poly = [(coord[0], coord[1]) for coord in path.vertices]
            ImageDraw.Draw(img).polygon(poly, fill=1)
        mask = np.array(img)
        patches = PatchCollection(patch_list, match_original=True)
        # paths = patches.get_paths()
        return mask


############
# Run Code #
############
if __name__ == '__main__':
    dstl = Train_DSTL()
    image, masks = dstl.display_three_band()
    # Select training set:
    training_set = image[masks[0]]
