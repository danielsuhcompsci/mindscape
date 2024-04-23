import os.path
import numpy as np
from foldrpp import Foldrpp

def brainVoxels(category, sourceDir, numVoxels, debugFlag): #"category" should hold the COCO category the data of which is being analyzed. sourceDir should hold the path to the directory containing the FOLDdata folder.
    str_attrs = []
    for index in range(1, numVoxels+1):
        str_attrs.append(f'voxel-{index}')
    if(debugFlag):
        print("str_attrs = ", str_attrs)
    num_attrs = [] #this was the convention used in all the other FOLD-RM datasets
    model = Foldrpp(str_attrs, num_attrs, label=category, pos_val='1')
    data = model.load_data(sourceDir, debugFlag=debugFlag)
    # if(debugFlag):
        # print(data)#for debugging
    return model, data

# def acute():
#     str_attrs = ['a2', 'a3', 'a4', 'a5', 'a6']
#     num_attrs = ['a1']
#     label, pos_val = 'label', 'yes'
#     model = Foldrpp(str_attrs, num_attrs, label, pos_val)
#     data = model.load_data('data/acute/acute.csv')
#     print('\n% acute dataset', len(data), len(str_attrs + num_attrs) + 1)
#     return model, data
