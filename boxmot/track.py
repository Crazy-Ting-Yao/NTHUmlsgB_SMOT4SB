import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
from boxmot import BoostTrack,OcSort,ByteTrack,StrongSort, DeepOcSort
import sys
import json
import os

if len(sys.argv) < 2:
    print("請提供資料夾名稱")
    sys.exit(1)
FILE_NAME = sys.argv[1]
frame_to_id = dict()

coco_json_path = f"/media/Pluto/huangtingyao/results/results_{FILE_NAME}.bbox.json"
#coco_json_path = f"/media/Pluto/yanfu/results/results_{FILE_NAME}.bbox.json"
#coco_json_path = f"/media/Pluto/huangtingyao/detection_results/{FILE_NAME}_train_results.json"
#coco_json_path = f"/home/jingxunlin/SMOT/dataset/results/results_{FILE_NAME}.bbox.json"
with open(coco_json_path, 'r') as f:
    coco_data = json.load(f)
for i, bbox in enumerate(coco_data):
    if bbox["image_id"] not in frame_to_id.keys():
        frame_to_id[bbox["image_id"]] = i

def get_bboxes_for_frame(frame_id):
    
    bboxes = []
    for i in range(10000):
        if frame_to_id[frame_id]+i >= len(coco_data):
            break
        if coco_data[frame_to_id[frame_id]+i]["image_id"] > frame_id:
            break
        bboxes.append(coco_data[frame_to_id[frame_id]+i]["bbox"])
        bboxes[-1].append(coco_data[frame_to_id[frame_id]+i]["score"])

    if len(bboxes) == 0:
        print(f"Frame ID {frame_id} 不存在")
        return []
    
    #bboxes = [ann["bbox"] for ann in coco_data["annotations"] if ann["image_id"] == image_id_map[(video_id, frame_id)]]
    xyxy = [[b[0], b[1], b[0]+b[2], b[1]+b[3], b[4]] for b in bboxes]
    #labels = [ann["track_id"] for ann in coco_data["annotations"] if ann["image_id"] == image_id_map[(video_id, frame_id)]]
    return xyxy


def create_video_from_images(image_folder, output_video_path, frame_rate=25):
    # define valid extension
    valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    
    # get all image files in the folder
    image_files = [f for f in os.listdir(image_folder) 
                   if os.path.splitext(f)[1] in valid_extensions]
    image_files.sort()  # sort the files in alphabetical order
    print(image_files)
    if not image_files:
        raise ValueError("No valid image files found in the specified folder.")
    
    # load the first image to get the dimensions of the video
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape
    
    # create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # codec for saving the video
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    
    # write each image to the video
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)
    
    # source release
    video_writer.release()
    print(f"Video saved at {output_video_path}")


video_dir = f"/home/jingxunlin/SMOT/dataset/SMOT4SB/train/{FILE_NAME}/"
# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# Load a pre-trained Faster R-CNN model
device = torch.device('cuda:0')  # Use 'cuda' if you have a GPU


tracker = ByteTrack(
    min_conf = 0.1      ,    # 保留幾乎所有框（避免翻翅導致 detector conf 掉到 0.1 以下就被砍）
    track_thresh = 0.15  ,    # 第一輪配對的主追蹤框，只取高信心
    match_thresh = 0.8   ,    # IOU 門檻調高，容忍外觀變形（例如展翅）
    track_buffer = 90    ,    # 最大容忍幾幀消失，90 幀約等於 3 秒（如果 30fps）
    frame_rate = 30      ,    # 根據你的影片設定，這樣 `track_buffer` 更準
    per_class = False,         # 如果鳥類只有一種，可以設 False 更穩定
)


save_dir = './results/'
sum = 0
from tqdm import tqdm
output_file = f"./val/{FILE_NAME}.txt"
dets = []
os.system(f"rm ./results/*")
with open(output_file, "w") as f:
    for frame_idx in tqdm(range(1, len(frame_names)+1)):
        # Capture frame-by-frame
        frame = cv2.imread(video_dir + frame_names[frame_idx-1])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert frame to tensor and move to device
        frame_tensor = torchvision.transforms.functional.to_tensor(frame).to(device)

        detections = get_bboxes_for_frame(frame_idx)
        
        dets = []
        for bbox in detections:
            x_min, y_min, x_max, y_max, conf = bbox
            dets.append([x_min, y_min, x_max, y_max, conf, 1.0])
        # Convert detections to numpy array (N X (x, y, x, y, conf, cls))
        dets = np.array(dets)
        
        # Update the tracker
        res = tracker.update(dets, frame)  # --> M X (x, y, x, y, id, conf, cls, ind)
        for obj in res:
            x_min, y_min, x_max, y_max, id, conf, clss, ind = obj
            bbox_width, bbox_height = x_max - x_min + 1, y_max - y_min + 1

            if bbox_width<=1 or bbox_height<=1:
                continue
            f.write(f"{frame_idx},{int(id)},{x_min},{y_min},{bbox_width},{bbox_height},1,1,1\n")
        # Plot tracking results on the image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tracker.plot_results(frame, show_trajectories=True)
        
        cv2.imwrite(os.path.join(save_dir, f"annotated_frame_{frame_idx:05d}.jpg"), frame)
create_video_from_images(save_dir, "output.mp4")
