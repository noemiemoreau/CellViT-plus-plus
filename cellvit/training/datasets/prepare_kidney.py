import os
import random

import matplotlib.pyplot as plt
from skimage.draw import polygon
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import json
import csv

TYPE_NUCLEI_DICT = {
    1: "Opal_480",
    2: "Opal_520",
    3: "Opal_570",
    4: "Opal_620",
    5: "Opal_690",
    6: "Outside",
    7: "Unclassified"
}

TRAIN_VAL_DICT = {
    "train" : ["A_hNiere_S3", "E_hNiere_S3", "J_hNiere_S1", "K_hNiere_S1"],
    "val" : ["C_hNiere_S3"]
}

if __name__ == "__main__":
    WSIs_path = "/scratch/nmoreau/CellViT_2025/kidney_data_256_40x/train/WSIs/"
    GTs_geojson_path = "/scratch/nmoreau/CellViT_2025/kidney_data_256_40x/train/GTs_geojson/"
    ROIs_geojson_path = "/scratch/nmoreau/CellViT_2025/kidney_data_256_40x/train/ROIs_geojson/"
    images_path = "/scratch/nmoreau/CellViT_2025/kidney_data_256_40x/train/images/"
    labels_path = "/scratch/nmoreau/CellViT_2025/kidney_data_256_40x/train/labels/"
    TYPE_NUCLEI_DICT_inv = {TYPE_NUCLEI_DICT[k]: k for k in TYPE_NUCLEI_DICT.keys()}
    patch_size = (256, 256)
    train_list = []
    val_list = []
    for image_name in os.listdir(WSIs_path):
        if not image_name.startswith("."):
            image_name = image_name[:-8]
            print(image_name)

            with open(GTs_geojson_path + image_name + ".geojson", 'r') as f:
                gson_cells_gt = json.load(f)
            with open(ROIs_geojson_path + image_name + ".geojson", 'r') as f:
                gson_rois_gt = json.load(f)
            rois_list = gson_rois_gt["features"]
            cells_gt_list = gson_cells_gt["features"]

            WSI_pil = Image.open(WSIs_path + image_name + "_PAS.png")
            WSI_array = np.array(WSI_pil)

            for roi in rois_list:
                roi_id = roi["id"]

                x_roi_list = [coord[0] for coord in roi["geometry"]["coordinates"][0]]
                y_roi_list = [coord[1] for coord in roi["geometry"]["coordinates"][0]]
                xmax = max(x_roi_list)
                xmin = min(x_roi_list)
                ymax = max(y_roi_list)
                ymin = min(y_roi_list)

                WSI_roi = WSI_array[ymin:ymax, xmin:xmax, :3]
                GT_type_map = np.zeros((ymax - ymin, xmax - xmin), dtype=np.uint16)
                GT_inst_map = np.zeros((ymax - ymin, xmax - xmin), dtype=np.uint16)
                i = 0
                for cell in cells_gt_list:
                    if cell["properties"]["objectType"] == "cell":
                        id_cell = cell["id"]
                        list_coord_cell = cell["geometry"]["coordinates"][0]
                        properties = cell["properties"]
                        if "classification" in properties.keys():
                            name = properties["classification"]["name"]
                        else:
                            name = "Outside"
                        x1 = list_coord_cell[0][0]
                        y1 = list_coord_cell[0][1]
                        if xmin < x1 < xmax and ymin < y1 < ymax:
                            i += 1
                            new_list_coord_cell = []
                            new_list_coord_nuclear = []
                            for coord in list_coord_cell:
                                new_coord_cell = [coord[0] - xmin, coord[1] - ymin]
                                new_list_coord_cell.append(new_coord_cell)
                            poly = np.array(new_list_coord_cell[:-1])
                            rr, cc = polygon(poly[:, 0], poly[:, 1], (GT_inst_map.shape[1], GT_inst_map.shape[0]))
                            GT_inst_map[cc, rr] = i
                            GT_type_map[cc, rr] = TYPE_NUCLEI_DICT_inv[name]
                path_number = 0
                for x in range(0, WSI_roi.shape[0], patch_size[0]):
                    for y in range(0, WSI_roi.shape[1], patch_size[1]):
                        path_number += 1
                        WSI_patch = WSI_roi[x:x + patch_size[0], y:y + patch_size[1]]
                        GT_inst_map_patch = GT_inst_map[x:x + patch_size[0], y:y + patch_size[1]]
                        GT_type_map_patch = GT_type_map[x:x + patch_size[0], y:y + patch_size[1]]
                        # rand = random.randrange(5)
                        rand = 0
                        if GT_inst_map_patch.shape == patch_size and rand == 0:
                            # print(WSI_patch.shape)
                            WSI_patch_pil = Image.fromarray(WSI_patch)
                            WSI_patch_pil.save(images_path + image_name + "_" + str(roi_id) + "_" + str(path_number) + ".png")

                            outdict = {"inst_map": GT_inst_map_patch, "type_map": GT_type_map_patch}
                            np.save(labels_path + image_name + "_" + str(roi_id) + "_" + str(path_number) + ".npy", outdict)
                            if image_name in TRAIN_VAL_DICT["train"]:
                                train_list.add([image_name + "_" + str(roi_id) + "_" + str(path_number)])
                            elif image_name in TRAIN_VAL_DICT["val"]:
                                val_list.add([image_name + "_" + str(roi_id) + "_" + str(path_number)])
                            # plt.imshow(WSI_patch)
                            # plt.show()
                            # plt.imshow(GT_inst_map_patch)
                            # plt.show()
                            # plt.imshow(GT_type_map_patch)
                            # plt.show()
                # plt.imshow(WSI_roi)
                # plt.show()
                # plt.imshow(GT_inst_map)
                # plt.show()
                # plt.imshow(GT_type_map)
                # plt.show()
    with open("/scratch/nmoreau/CellViT_2025/kidney_data_256_40x/splits/fold_0/train.csv", 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(train_list)
    with open("/scratch/nmoreau/CellViT_2025/kidney_data_256_40x/splits/fold_0/val.csv", 'w') as f:
        csv_writer = csv.writer(val_list)


