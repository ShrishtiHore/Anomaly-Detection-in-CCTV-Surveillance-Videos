import os
import subprocess


def pose_estimation(outpath, outanns):

    max_length = 2e03
    for i,(dire, folds, fils) in enumerate(os.walk(outpath)):
        if i ==0:
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
            subprocess.call(cmd , shell=True)