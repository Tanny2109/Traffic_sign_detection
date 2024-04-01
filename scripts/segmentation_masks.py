# import fiftyone as fo
# import fiftyone.brain as fob
# from fiftyone import ViewField as F
import os
import cv2
import wget
import matplotlib.pyplot as plt
from zipfile import ZipFile
import torch
import torchvision
from ultralytics import YOLO
import numpy as np
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import PIL

cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)

model_name = "/home/GTL/tsutar/Traffic_sign_detection/scripts/runs/detect/train/weights/2024-14-03-best.pt"
model = YOLO(model_name)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "default"
CHECKPOINT_PATH = "/home/GTL/tsutar/Traffic_sign_detection/SAM_chks/sam_vit_h_4b8939.pth"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_predictor = SamPredictor(sam)

img_dir = parent_dir + "/dataset/consolidated_dataset/images"
masked_img_dir = "/home/GTL/tsutar/Traffic_sign_detection/dataset/consolidated_dataset/masked_image"

img_idx=0

for img in os.listdir(img_dir):

    if img_idx == 1:
        break

    # predict = model(img)[0].boxes.xyxy

    # transformed_boxes = mask_predictor.transform.apply_boxes_torch(predict, img.shape[:2])
    plt.imshow(img)
    plt.show()

    

    # mask_predictor.set_image(img)
    # masks, scores, logits = mask_predictor.predict_torch(
    #     boxes = transformed_boxes,
    #     multimask_output=False,
    #     point_coords=None,
    #     point_labels=None
    # )

    # # combine all masks into one for easy visualization
    # final_mask = None
    # for i in range(len(masks) - 1):
    #     if final_mask is None:
    #         final_mask = np.bitwise_or(masks[i][0], masks[i+1][0])
    #     else:
    #         final_mask = np.bitwise_or(final_mask, masks[i+1][0])

    # visualize the predicted masks
    # plt.figure(figsize=(10, 10))
    # plt.imshow(img)
    # # plt.imshow(final_mask, cmap='gray', alpha=0.7)
    # plt.show()

    img_idx += 1