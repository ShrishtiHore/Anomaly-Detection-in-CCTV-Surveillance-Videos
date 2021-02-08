import os
import subprocess
import argparse

from TimeSeries.split_vid import vid_into_frames


parser = argparse.ArgumentParser('Pose estimation on ')
parser.add_argument('--split', action='store_true', default=False, help='Split videos')
parser.add_argument('--pose', action='store_true', default=False, help='Pose estimation')






def pose_estimation(outpath, outanns):
    max_length = 2e03
    args = parser.parse_args()
    for i, (dire, folds, fils) in enumerate(os.walk(outpath)):
        if i == 0:
            print('Main dataset directory: {}\n  Subdirectories within it: {}\n'.format(dire, folds))
            continue

        # run command on the frames
        if len(folds) == 0 and len(fils) > 0:

            # loop over video frames
            print('Processing video frames in: {}'.format(dire))

            # run a python module and output the json file at the correpondent annotation folder
            ldir = dire.split('/')
            out_json = os.path.join(outanns, ldir[-2], ldir[-1])
            if not os.path.isdir(out_json):
                os.makedirs(out_json)
            cmd = "python demo.py --indir {}  --outdir {}".format(dire, out_json)
            subprocess.call(cmd, shell=True)









def main():
    # root directory
    root_direct = '/home/ubuntu/Avenue/AvenueDataset'
    # UCF_Anomalies dataset path
    dset_root = os.path.join(root_direct, 'training_videos')
    args = parser.parse_args()

    out_split = os.path.join(root_direct, 'train')
    if not os.path.isdir(out_split):
        os.mkdir(out_split)

    # create new directory structure and write frames at path out_split
    if args.split:
        vid_into_frames(dset_root, out_split)


    # output annotation path
    outanns = os.path.join(root_direct, 'train_ann')
    if not os.path.isdir(outanns):
        os.mkdir(outanns)

    # set working directory for AlphaPose framework
    os.chdir('/home/ubuntu/PoseData/AlphaPose/')
    # create new directory structure and write annotation files at out_ann
    if args.pose:
        print("###############################")
        print('Running pose estimation: \n')
        pose_estimation(out_split, outanns)

if __name__ == "__main__":
    main()
