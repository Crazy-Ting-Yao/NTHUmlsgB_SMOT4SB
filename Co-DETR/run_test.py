import subprocess
from pathlib import Path

config_path = 'projects/configs/co_deformable_detr/co_deformable_detr_swin_large_900q_3x_smot4sb.py'
checkpoint_path = 'model/epoch_12.pth'
output_dir = Path('results_12')
output_dir.mkdir(parents=True, exist_ok=True)

root_img_dir = Path('dataset/SMOT4SB/private_test')
ann_dir = Path('dataset/SMOT4SB/private_test_anns')

for folder in sorted(root_img_dir.iterdir()):
    if not folder.is_dir():
        continue

    folder_name = folder.name
    ann_file = ann_dir / f'{folder_name}.json'
    img_prefix = folder

    json_prefix = output_dir / f'results_{folder_name}'

    command = [
        'python', 'tools/test.py', config_path,
        checkpoint_path,
        '--format-only',
        '--cfg-options',
        f'data.test.ann_file={ann_file}',
        f'data.test.img_prefix={img_prefix}',
        '--eval-options',
        f'jsonfile_prefix={json_prefix}',
    ]

    print(f'Running inference for folder {folder_name}...')
    subprocess.run(command)

print("All inference tasks completed.")
