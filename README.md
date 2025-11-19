# PTTSafe_Artery_Hypertune
Physiology-Constrained Multi-Modal Signal Synchronization and Refinement for Cuffless Continuous Blood Pressure Estimation
Physiological PTT-Preserved ECG-PPG Synchronization and Industrial-Grade Preprocessing Enable Cuffless Blood Pressure Estimation with Intra-Arterial Accuracy



<img width="3022" height="1719" alt="new_final_1_seg30" src="https://github.com/user-attachments/assets/eeed87ea-8e95-4208-9931-2cc1b0140c59" />

conda create -n drink python=3.10 -y

conda activate drink

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1  

pip install numpy==1.23.5 pandas openpyxl scikit-learn tqdm matplotlib seaborn pyPPG==1.0.14 neurokit2 dotmap scipy

data:
https://huggingface.co/datasets/peter962/PTTSafe_Artery_Hypertune

cmd
python show.py --fold 1 --num_normal 30

python PTTSafe_Artery_Hypertune.py --cv
