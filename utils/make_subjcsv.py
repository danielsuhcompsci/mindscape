import numpy as np
from pycocotools.coco import COCO
import pandas as pd
from pandas import json_normalize
import os.path
import argparse
import nilearn.image

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--subject', type=int, default=None, help='subject for which csv is made')
parser.add_argument('-n', '--nsddir', type=str, default=None, help='source directory of nsd data')
parser.add_argument('-o', '--output', type=str, default='.', help='destination directory')
opt = parser.parse_args()

subject = opt.subject
nsd_dir = opt.nsddir
out_dir = opt.output
betas = np.load(os.path.join(nsd_dir, f'nsddata_betas/masked/subj{subject:02n}.npy'))
prf_atlas = nilearn.image.get_data(os.path.join(nsd_dir, f'nsddata/ppdata/subj{subject:02n}/func1pt8mm/roi/prf-visualrois.nii.gz'))
floc_atlas = nilearn.image.get_data(os.path.join(nsd_dir, f'nsddata/ppdata/subj{subject:02n}/func1pt8mm/roi/floc-faces.nii.gz'))
headers = []


for i in range(betas.shape[0]):
    threshold = np.percentile(betas[i], 95)
    betas[i] = betas[i] > threshold

for i in range(np.sum(prf_atlas == 1)):
    headers = np.concatenate((headers, [f'V1v-{i}']))
for i in range(np.sum(prf_atlas == 2)):
    headers = np.concatenate((headers, [f'V1d-{i}']))
for i in range(np.sum(prf_atlas == 3)):
    headers = np.concatenate((headers, [f'V2v-{i}']))
for i in range(np.sum(prf_atlas == 4)):
    headers = np.concatenate((headers, [f'V2d-{i}']))
for i in range(np.sum(prf_atlas == 5)):
    headers = np.concatenate((headers, [f'V3v-{i}']))
for i in range(np.sum(prf_atlas == 6)):
    headers = np.concatenate((headers, [f'V3d-{i}']))
for i in range(np.sum(floc_atlas == 1)):
    headers = np.concatenate((headers, [f'OFA-{i}']))
for i in range(np.sum(floc_atlas == 2)):
    headers = np.concatenate((headers, [f'FFA_1-{i}']))
for i in range(np.sum(floc_atlas == 3)):
    headers = np.concatenate((headers, [f'FFA_2-{i}']))
for i in range(np.sum(floc_atlas == 4)):
    headers = np.concatenate((headers, [f'mTL_faces-{i}']))
for i in range(np.sum(floc_atlas == 5)):
    headers = np.concatenate((headers, [f'aTL_faces-{i}']))

csv = pd.DataFrame(np.asarray(betas, dtype=np.int32), columns=headers)

viewedImgIds = pd.read_csv(os.path.join(
    nsd_dir, f'nsddata/ppdata/subj{subject:02n}/behav/responses.tsv'), delimiter='\t', usecols=[4])['73KID']
viewedImgIds = (viewedImgIds - 1).transform(pd.read_csv(os.path.join(
    nsd_dir, 'nsddata/experiments/nsd/nsd_stim_info_merged.csv'), usecols=[1])['cocoId'].at.__getitem__)

cocoval = COCO(os.path.join(nsd_dir, 'nsddata_stimuli/stimuli/nsd/annotations/instances_val2017.json'))
cocotrain = COCO(os.path.join(nsd_dir, 'nsddata_stimuli/stimuli/nsd/annotations/instances_train2017.json'))
cat_ids = cocoval.getCatIds()
categories = json_normalize(cocoval.loadCats(cat_ids))

for cat_id in cat_ids:
    catImgIds = np.concatenate((cocoval.getImgIds(catIds=[cat_id]), cocotrain.getImgIds(catIds=[cat_id])))
    csv.insert(csv.shape[1], categories[categories['id']==cat_id]['name'].iloc[0], viewedImgIds.isin(catImgIds) * 1)

csv.insert(csv.shape[1], 'x-coord', np.random.randint(2, size=(betas.shape[0],1)))
csv.insert(csv.shape[1], 'y-coord', np.random.randint(2, size=(betas.shape[0],1)))

csv.to_csv(os.path.join(out_dir, f'subj{subject:02n}.csv'))