# Core libraries
import os
import sys

sys.path.append(os.path.abspath("../"))
import cv2
import json
import math
import numpy as np
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# My libraries
from Utilities.DataUtils import DataUtils
from Datasets.RGBDCows2020_will import RGBDCows2020
import matplotlib.patches as patches

class DatasetVisualiser(object):
    """
    Class for visualising stats/graphs regarding a dataset
    """

    def __init__(self):
        """
        Class constructor
        """

        pass

    """
    Public methods
    """

    """
    (Effectively) private methods
    """

    """
    Staticmethods
    """

    @staticmethod
    def binsLabels(bins, **kwargs):
        bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
        plt.xticks(np.arange(min(bins) + bin_w / 2, max(bins), bin_w), bins, **kwargs)
        plt.xlim(bins[0], bins[-1])

    @staticmethod
    def tileCategories(dataset, max_cols=20, img_type="RGB", random=True):
        """ Produce a tiled image of all categories

        Retrieve a random instance from each category and display them all in a grid of tiled images.
        """

        # Load up the file
        with open(dataset.getSplitsFilepath()) as handle:
            random_splits = json.load(handle)
            print(f"Retrieved splits at {dataset.getSplitsFilepath()}")

        # Which fold shall we use
        if random:
            fold = random.choice(list(random_splits.keys()))
        else:
            fold = str('001')

        # Retrieve a random example for each category
        if random:
            examples = {cat: random.choice(random_splits[fold][cat]['train']) for cat in random_splits[fold].keys()}

        else:
            examples = {cat: random_splits[cat]['train'][0] for cat in random_splits.keys()}
            #examples = {cat: random_splits[fold][cat]['train'][0] for cat in random_splits[fold].keys()}


        # Get the root directory for the dataset
        root_dir = os.path.join(dataset.getDatasetPath(), img_type)

        # Actually load the images into memory
        example_images = {cat: cv2.imread(os.path.join(root_dir, cat, filename)) for cat, filename in examples.items()}

        # Retrieve the number of classes we have
        num_classes = dataset.getNumClasses()
        assert num_classes == len(example_images.keys())

        # Compute the number of rows needed
        nrows = int(math.ceil(num_classes / max_cols))

        # Create the figure
        fig = plt.figure(figsize=(40, 9))
        grid = ImageGrid(fig, 111, nrows_ncols=(nrows, max_cols), axes_pad=0.1)

        # Go through and turn each axes visibility off
        for ax in grid:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.axis('off')

        count = 0
        # Iterate through each category in order
        for ax, cat in zip(grid, sorted(example_images.keys())):
            # Get the numpy image
            np_img = example_images[cat]

            # Convert to RGB (from BGR)
            np_img = np_img[..., ::-1].copy()

            # Convert to PIL
            pil_img = Image.fromarray(np_img, "RGB")

            # Assign the image to this axis
            ax.imshow(pil_img)
            if 'train' in cat :  # extra train data
                rect = patches.Rectangle((0, 0), 490, 240, linewidth=8, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            if cat in ['020','157','165','185']:
                rect = patches.Rectangle((0, 0), 490, 240, linewidth=8, edgecolor='b', facecolor='none')
                ax.add_patch(rect)
        plt.tight_layout()
        plt.savefig(f"category-examples.png")
        plt.show()



# Entry method/unit testing method
if __name__ == '__main__':
    # Visualise examples of the dataset
    dataset = RGBDCows2020(suppress_info=True)

    # can run  images for each folder
    DatasetVisualiser.tileCategories(dataset, img_type="RGB", random=False)  # RGB  Depth