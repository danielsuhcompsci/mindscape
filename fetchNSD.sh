#!/bin/bash

if [ $OPTIND -eq 0 ]; then
    fetchAtlas=true
    fetchBetas=true
    fetchCoco=true
    fetchExpData=true
    fetchImages=true
else
    fetchAtlas=false
    fetchBetas=false
    fetchCoco=false
    fetchExpData=false
    fetchImages=false
fi

while getopts 'abcehi' opt; do
    case $opt in
        a)
            fetchAtlas=true
            ;;

        b)
            fetchBetas=true
            ;;

        c)
            fetchCoco=true
            ;;

        e)
            fetchExpData=true
            ;;

        i)
            fetchImages=true
            ;;

        ?)
            printf "Usage $(basename $0) [-a] [-b] [-c] [-e] [-i]\n -a\tDownload atlases\n -b\tDownload betas\n -c\tDownload coco info\n -e\tDownload experiment info\n -i\tDownload images\n"
            exit 1
            ;;
    esac
done

mkdir -p nsd
cd nsd


if $fetchExpData; then
    aws s3 cp s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.csv nsddata/experiments/nsd/
    aws s3 cp s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_expdesign.mat nsddata/experiments/nsd/
fi

if $fetchAtlas; then
    aws s3 cp s3://natural-scenes-dataset/nsddata/freesurfer/subj01/label/prf-visualrois.mgz.ctab nsddata/freesurfer/fsaverage/label/
    aws s3 cp s3://natural-scenes-dataset/nsddata/freesurfer/subj01/label/floc-faces.mgz.ctab nsddata/freesurfer/fsaverage/label/
fi

if $fetchImages; then
    aws s3 cp s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 nsddata_stimuli/stimuli/nsd/
fi

if $fetchCoco; then
    #all image annotations
    curl http://images.cocodataset.org/annotations/annotations_trainval2017.zip --output coco_annotations.zip
    unzip  -o coco_annotations.zip -d nsddata_stimuli/stimuli/nsd/
    rm coco_annotations.zip
fi

for ((i = 1; i <= 8; i++)); do

    if $fetchExpData; then
        aws s3 cp s3://natural-scenes-dataset/nsddata/ppdata/subj0$i/behav/responses.tsv nsddata/ppdata/subj0$i/behav/
    fi

    if $fetchAtlas; then
        aws s3 cp s3://natural-scenes-dataset/nsddata/ppdata/subj0$i/func1pt8mm/roi/prf-visualrois.nii.gz nsddata/ppdata/subj0$i/func1pt8mm/roi/
        aws s3 cp s3://natural-scenes-dataset/nsddata/ppdata/subj0$i/func1pt8mm/roi/floc-faces.nii.gz nsddata/ppdata/subj0$i/func1pt8mm/roi/
    fi

    if $fetchBetas; then
        for ((j = 1; j <= 40; j++)); do
            fileName=$(printf "s3://natural-scenes-dataset/nsddata_betas/ppdata/subj%02d/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session%02d.nii.gz" $i $j)
            # if file doesn't exist, move on to next subject 
            if [ "$(aws s3 ls $fileName)" == "" ]; then
                break
            fi
            aws s3 cp $fileName nsddata_betas/ppdata/subj0$i/func1pt8mm/betas_fithrf_GLMdenoise_RR/
        done
    fi
   done
