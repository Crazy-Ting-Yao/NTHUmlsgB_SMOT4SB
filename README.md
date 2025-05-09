# NTHUmlsgB_SMOT4SB

## 0. CUDA Requirement
Please use CUDA 11.3.

## 1. Environment Preparation
### 1.1 Create Conda Environment
```bash
git clone https://github.com/Crazy-Ting-Yao/NTHUmlsgB_SMOT4SB.git
conda create --name boxmot python=3.10 -y
conda activate boxmot
pip install boxmot
conda deactivate

conda create --name codetr python=3.8 -y
conda activate codetr
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
cd NTHUmlsgB_SMOT4SB/Co-DETR
pip install -r requirements.txt
pip install yapf==0.40.1
pip install pillow
pip install -v -e .
```

### 1.2 Preparing Private Test Data
Place the SMOT4SB dataset under `NTHUmlsgB_SMOT4SB/Co-DETR/dataset`. The directory structure should look like this:

```bash
dataset
└ SMOT4SB
　 └ private_test
```

### 1.3 Preparing Model

Download model from [here](https://drive.google.com/file/d/1U0FqjEUv9-K7w7vhpXGPdDkOiVZzgE04/view?usp=sharing) and place it under `NTHUmlsgB_SMOT4SB/Co-DETR/model`. The directory structure should look like this:

```bash
model
└ epoch_12.pth
```

## 2. Run the code

```
python annotations_generator.py
python run_test.py
cd ../boxmot
conda activate boxmot
./infer.sh
```
