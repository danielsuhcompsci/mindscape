#!/bin/bash
mkdir -p nsd
cd nsd

mkdir -p nsddata/experiments/nsd
mkdir -p nsddata/freesurfer/fsaverage/label/
mkdir -p nsddata_stimuli/stimuli/nsd/

#information about every image
aws s3 cp s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.csv nsddata/experiments/nsd/

#information about experiment design
aws s3 cp s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_expdesign.mat nsddata/experiments/nsd/

#atlas
aws s3 cp s3://natural-scenes-dataset/nsddata/freesurfer/fsaverage/label/streams.mgz.ctab nsddata/freesurfer/fsaverage/label/

#all images
aws s3 cp s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 nsddata_stimuli/stimuli/nsd/

#all image annotations
curl http://images.cocodataset.org/annotations/annotations_trainval2017.zip --output coco_annotations.zip
unzip  -o coco_annotations.zip -d nsddata_stimuli/stimuli/nsd/
rm coco_annotations.zip


for ((i = 1; i <= 8; i++)); do
    mkdir -p nsddata/ppdata/subj0$i/behav
    mkdir -p nsddata/ppdata/subj0$i/func1pt8mm/roi
    mkdir -p nsddata_betas/ppdata/subj0$i/func1pt8mm/betas_fithrf_GLMdenoise_RR/

    #information about images shown to subject
    aws s3 cp s3://natural-scenes-dataset/nsddata/ppdata/subj0$i/behav/responses.tsv nsddata/ppdata/subj0$i/behav/

    #atlas of regions of interest
    aws s3 cp s3://natural-scenes-dataset/nsddata/ppdata/subj0$i/func1pt8mm/roi/streams.nii.gz nsddata/ppdata/subj0$i/func1pt8mm/roi/
    
    for ((j = 0; j <= 3; j++)); do
        for ((k = 1; k <= 9; k++)); do
            # if file doesn't exist, move on to next subject 
            if [ "$(aws s3 ls s3://natural-scenes-dataset/nsddata_betas/ppdata/subj0${i}/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session${j}${k}.nii.gz)" == "" ]; then
                break 2
            fi
            aws s3 cp s3://natural-scenes-dataset/nsddata_betas/ppdata/subj0$i/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session$j$k.nii.gz nsddata_betas/ppdata/subj0$i/func1pt8mm/betas_fithrf_GLMdenoise_RR/
        done
        aws s3 cp s3://natural-scenes-dataset/nsddata_betas/ppdata/subj0$i/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session$((j+1))0.nii.gz nsddata_betas/ppdata/subj0$i/func1pt8mm/betas_fithrf_GLMdenoise_RR/
    done
done