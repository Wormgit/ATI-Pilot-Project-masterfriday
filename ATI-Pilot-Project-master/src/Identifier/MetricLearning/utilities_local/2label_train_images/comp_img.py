# Core libraries
import os
import sys
import cv2
import json, math
import csv
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image

# out put name: 014_0_179_650_
#               folder +subfolder, will's label

csv_path = '/home/io18230/Desktop/correct.csv'
pre_img_path = "/home/io18230/Desktop/RGBDCows202066/Identification301_4days/RGB"
gt_img_path ='/home/io18230/Desktop/RGBDCows2020/Identification_WILL/RGB'
lastname5= '001/0'
corrected_pt='/home/io18230/Desktop/correct.csv'
corrected = 1 # 1 is correction mode
# run in 10 min and report an error.



examples_t = []
count_t = 0
show_number_imag= 4
max_cols=show_number_imag
folder = []
precheck_gt= []
plot = 1

def plottt(example_images):
    # Compute the number of rows needed
    num_classes = 20
    nrows = int(math.ceil(num_classes / max_cols))

    # Create the figure
    fig = plt.figure(figsize=(9,3))
    grid = ImageGrid(fig, 111, nrows_ncols=(nrows, max_cols), axes_pad=0.1)

    # Go through and turn each axes visibility off
    for ax in grid:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.axis('off')

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

    plt.tight_layout()


if not corrected:
    csvFile = csv.reader(open(csv_path, 'r'), delimiter=',')
    reader = list(csvFile)
    del reader[0]
    for i in range(len(reader)):
        test_image = os.path.join(pre_img_path, reader[i][1])

        # 5 test images
        if count_t < show_number_imag:
            examples_t.append(test_image)
            count_t += 1

        name5 = reader[i][1][:5]
        # next tracklet
        if name5!= lastname5:

            # 4 predict gt images
            pre = "%03d" % (int(float(reader[i-1][2])))
            pre_gt_folder = os.path.join(gt_img_path, pre)

            examples = []
            count = 0
            for items in os.listdir(pre_gt_folder):
                examples.append(items)
                count += 1
                if count == show_number_imag:
                    break

            if plot:
                # Actually load the images into memory
                example_images_t = {cat: cv2.imread(filename) for cat, filename in enumerate(examples_t)}
                example_images_g = {cat+show_number_imag: cv2.imread(os.path.join(pre_gt_folder, filename)) for cat, filename in
                                  enumerate(examples)}
                example_images = {}
                example_images.update(example_images_t)
                example_images.update(example_images_g)

                plottt(example_images)

                plt.savefig(f"/home/io18230/Desktop/ioo/all/{lastname5[0:3]}_{lastname5[4]}_{pre}_{i}_.png")
                plt.close()

            folder.append(lastname5)
            precheck_gt.append(pre)

            count_t = 0
            examples_t = []

        lastname5 = name5

    data1 = pd.DataFrame({'folder': folder, 'precheck_gt': precheck_gt})
    data1.to_csv("/home/io18230/Desktop/ioo/" + 'doublechecked.csv')


else:

    csvFile = csv.reader(open(corrected_pt, 'r'), delimiter=',')
    reader = list(csvFile)
    del reader[0]
    for i in range(len(reader)):
        lastname5 = reader[i][1]
        test_image=os.path.join(pre_img_path, reader[i][1])

        examples_t = []
        count = 0
        for items in os.listdir(test_image):
            examples_t.append(items)
            count += 1
            if count == show_number_imag:
                break


        # 4 predict gt images
        if reader[i][3] == '1':
            pre = "%03d" % (int(float(reader[i][4])))
        else:
            pre = "%03d" % (int(float(reader[i][2])))
        if pre == '054' or  pre == '179' : #totally black or the one less than 5 images (179)
            pre = '027'  # use a white to replace it. do not show to the user.
        pre_gt_folder = os.path.join(gt_img_path, pre)


        examples = []
        count = 0
        for items in os.listdir(pre_gt_folder):
            examples.append(items)
            count += 1
            if count == show_number_imag:
                break

        if plot:
            # Actually load the images into memory
            example_images_t = {cat: cv2.imread(os.path.join(test_image, filename)) for cat, filename in enumerate(examples_t)}
            example_images_g = {cat+show_number_imag: cv2.imread(os.path.join(pre_gt_folder, filename)) for cat, filename in
                              enumerate(examples)}
            example_images = {}
            example_images.update(example_images_t)
            example_images.update(example_images_g)

            plottt(example_images)
            if reader[i][3] == '1':
                plt.savefig(f"/home/io18230/Desktop/ioo/corrected/{reader[i][1][0:3]}_{reader[i][1][4]}_{reader[i][4]}.png")
            else:
                plt.savefig(
                    f"/home/io18230/Desktop/ioo/normal/{reader[i][1][0:3]}_{reader[i][1][4]}_{reader[i][4]}.png")
            plt.close()