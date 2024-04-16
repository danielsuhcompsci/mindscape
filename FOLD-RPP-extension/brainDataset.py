import os.path
import numpy as np
from foldrpp import Foldrpp

def brainVoxels(category, sourceDir, numVoxels, debugFlag): #"category" should hold the COCO category the data of which is being analyzed. sourceDir should hold the path to the directory containing the FOLDdata folder.
    #I'm not sure whether I should exclude the other category values from parsing, but I'll leave them in for structural completeness' sake
    str_attrs = []
    for index in range(1, numVoxels+1):
        str_attrs.append(f'voxel-{index}')
    str_attrs_2 = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    str_attrs.append(str_attrs_2)
    if(debugFlag):
        print(str_attrs)
    num_attrs = [] #this was the convention used in all the other FOLD-RM datasets
    model = Foldrpp(str_attrs, num_attrs, label=category, pos_val='1')
    data = model.load_data(sourceDir)
    if(debugFlag):
        print(data[0])#for debugging
    # print('\n% mindscape voxel dataset', len(data), len(str_attrs + num_attrs), np.shape(data))
    return model, data

# def acute():
#     str_attrs = ['a2', 'a3', 'a4', 'a5', 'a6']
#     num_attrs = ['a1']
#     label, pos_val = 'label', 'yes'
#     model = Foldrpp(str_attrs, num_attrs, label, pos_val)
#     data = model.load_data('data/acute/acute.csv')
#     print('\n% acute dataset', len(data), len(str_attrs + num_attrs) + 1)
#     return model, data
