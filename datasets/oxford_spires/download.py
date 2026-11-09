from huggingface_hub import hf_hub_download
repo='ori-drs/oxford_spires_dataset'
seq='sequences/2024-03-18-christ-church-01'
files=[
 'calibration/cam0.yaml','calibration/cam1.yaml','calibration/cam2.yaml','calibration/calibration_target.txt',
 seq+'/processed/trajectory/hba-tum.txt',
 seq+'/processed/trajectory/vilens-slam-tum.txt',
 seq+'/processed/trajectory/colmap-tum.txt',
 seq+'/raw/imu.csv',
 seq+'/raw/lidar-clouds.zip',
]
for f in files:
    try:
        p=hf_hub_download(repo_id=repo, filename=f, repo_type='dataset', local_dir='datasets/oxford_spires')
        print('ok', p, flush=True)
    except Exception as e:
        print('FAILED', f, e, flush=True)
print('ALL DONE', flush=True)
