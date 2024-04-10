import numpy as np
from pycocotools.coco import COCO
import pandas as pd
from pandas import json_normalize
import os.path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--subject', type=int, default=None, help='subject for which csv is made')
parser.add_argument('-n', '--nsddir', type=str, default=None, help='source directory of nsd data')
parser.add_argument('-o', '--output', type=str, default='.', help='destination directory')
opt = parser.parse_args()

subject = opt.subject
nsd_dir = opt.nsddir
out_dir = opt.o

cocoval = COCO(os.path.join(nsd_dir, 'nsddata_stimuli/stimuli/nsd/annotations/instances_val2017.json'))
cocotrain = COCO(os.path.join(nsd_dir, 'nsddata_stimuli/stimuli/nsd/annotations/instances_train2017.json'))
cat_ids = cocoval.getCatIds()
categories = json_normalize(cocoval.loadCats(cat_ids))

csv = pd.DataFrame(np.zeros((3000, 0)))
viewedImgIds = pd.read_csv(os.path.join(
    nsd_dir, f'nsddata/ppdata/subj{subject:02n}/behav/responses.tsv'), delimiter='\t', usecols=[4])['73KID']
viewedImgIds = (viewedImgIds - 1).transform(pd.read_csv(os.path.join(
    nsd_dir, 'nsddata/experiments/nsd/nsd_stim_info_merged.csv'), usecols=[1])['cocoId'].at.__getitem__)

for cat_id in cat_ids:
    catImgIds = np.concatenate(cocoval.getImgIds(catIds=[cat_id]), cocotrain.getImgIds(catIds=[cat_id]))
    csv.insert(csv.shape[1], categories[categories['id']==cat_id]['name'].iloc[0], viewedImgIds.isin(catImgIds) * 1)

#dummy values
for i in range(5246):
    csv.insert(csv.shape[1], 'voxel-{i}', np.random.randint(2, size=(3000, 1)))
csv.insert(csv.shape[1], 'x-coord', np.random.randint(2, size=(3000,1)))
csv.insert(csv.shape[1], 'y-coord', np.random.randint(2, size=(3000,1)))

csv.to_csv(os.path.join(out_dir, 'subj{subject:02n}.csv'))
