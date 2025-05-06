import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
from boxmot import BoostTrack,OcSort,ByteTrack,StrongSort
import sys
import json
import os
def compute_iou(boxA, boxB):
    """
    計算兩個 bounding box 的 IoU（Intersection over Union）

    Parameters:
    - box1, box2: [x1, y1, x2, y2]

    Returns:
    - IoU 值（float）
    """
    # 計算交集區域
    xA = (boxA[0] + boxA[2])/2
    yA = (boxA[3] + boxA[1])/2
    xB = (boxB[0] + boxB[2])/2
    yB = (boxB[3] + boxB[1])/2

    return np.sqrt((xA-xB)**2 + (yA-yB)**2)

frame_to_id = dict()

coco_json_path = "/media/Pluto/huangtingyao/MVATeam1/ensemble/intern_h_public_nosahi_randflip.json"
#coco_json_path = f"/media/Pluto/huangtingyao/MVATeam1/ensemble/cascade_nwd_paste_howard_0604.json"
gt_path = "../MVA2025-SMOT4SB/eval_inputs/ref/train/0054/gt/gt.txt"
with open(coco_json_path, 'r') as f:
    coco_data = json.load(f)

with open(gt_path, 'r') as f:
    gt = f.readlines()

score = 0.0
cnt = 0
for _ in gt:
    kk = list(map(float, _.split(',')[2:6]))
        
    kk[2] += kk[0]
    kk[3] += kk[1]
    max_iou = 0
    ttt = 0
    for i, bbox in enumerate(coco_data):
        bbox1 = bbox["bbox"]
        bbox1[2] += bbox1[0]
        bbox1[3] += bbox1[1]
        if compute_iou(bbox1, kk) > max_iou:
            max_iou = compute_iou(bbox1, kk)
            ttt = bbox["score"]
    print(ttt, max_iou)
    score += ttt
    cnt += 1

print(f"score = {score/cnt},   cnt={cnt}")
