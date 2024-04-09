import numpy as np
from pycocotools.coco import COCO
import pandas as pd
from pandas import json_normalize
import os.path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--subject", type=int, default=None)
parser.add_argument("--nsddir", type=str, default=None)
parser.add_argument("-o", type=str, default='.')
opt = parser.parse_args()

subject = opt.subject
nsd_dir = opt.nsddir
out_dir = opt.o
nsd_dir = '../../nsd'

coco = COCO(os.path.join(nsd_dir, 'nsddata_stimuli/stimuli/nsd/annotations/instances_train2017.json'))
cat_ids = coco.getCatIds()
categories = json_normalize(coco.loadCats(cat_ids))

csv = pd.DataFrame(np.zeros((3000, 0)))
viewedImgIds = pd.read_csv(os.path.join(
    nsd_dir, 'nsddata/ppdata/subj{:02n}/behav/responses.tsv'.format(subject)), 
    delimiter='\t', usecols=[4])['73KID']
viewedImgIds = (viewedImgIds - 1).transform(pd.read_csv(os.path.join(
    nsd_dir, 'nsddata/experiments/nsd/nsd_stim_info_merged.csv'), usecols=[1])['cocoId'].at.__getitem__)

for cat_id in cat_ids:
    catImgIds = coco.getImgIds(catIds=[cat_id])
    csv.insert(csv.shape[1], categories[categories['id']==cat_id]['name'].iloc[0], viewedImgIds.isin(catImgIds) * 1)
csv.to_csv(os.path.join(out_dir, 'subj{:02n}.csv'.format(subject)))