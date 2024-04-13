import numpy as np
import pandas as pd
import os.path
import argparse
import nilearn.image

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--subject', type=int, default=None, help='subject for which csv is made')
parser.add_argument('-n', '--nsddir', type=str, default=None, help='source directory of nsd data')
opt = parser.parse_args()

subject = opt.subject
nsd_dir = opt.nsddir
betas_dir = os.path.join(nsd_dir, f'nsddata_betas/ppdata/subj{subject:02n}/func1pt8mm/betas_fithrf_GLMdenoise_RR/')
out_dir = os.path.join(nsd_dir, 'nsddata_betas/masked')

prf_atlas = nilearn.image.get_data(os.path.join(nsd_dir, f'nsddata/ppdata/subj{subject:02n}/func1pt8mm/roi/prf-visualrois.nii.gz'))
floc_atlas = nilearn.image.get_data(os.path.join(nsd_dir, f'nsddata/ppdata/subj{subject:02n}/func1pt8mm/roi/floc-faces.nii.gz'))

maskedImages = np.empty((0, 5246))
for session in range(1, 41):
    session_file = os.path.join(betas_dir, f'betas_session{session:02n}.nii.gz')
    if not os.path.isfile(session_file):
        break
    session_image = nilearn.image.get_data(session_file)
    for trial in range(750):
        trial_image = session_image[:,:,:,trial][((prf_atlas > 0) & (prf_atlas < 7)) | (floc_atlas > 0)]
        maskedImages = np.concatenate((maskedImages, [trial_image]), 0)
os.makedirs(out_dir, exist_ok=True)
np.save(os.path.join(out_dir, f'subj{subject:02n}.npy'), maskedImages)