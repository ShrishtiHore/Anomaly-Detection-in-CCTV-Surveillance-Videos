import os
import subprocess


def vid_into_frames(root_dataset, out_path):

    for i, (dire, folds, fil) in enumerate(os.walk(root_dataset)):
        for vid_name in fil:

            # path to video
            vid_path = os.path.join(dire, vid_name)

            # create a folder for each video
            fname = vid_name.split('.avi')[0]
            path_fold = os.path.join(out_path, dire.split('/')[-1], fname)
            if not os.path.isdir(path_fold):
                os.makedirs(path_fold)

            # executing ffmpeg -i file.mp4 -vf fps=5 path/%04d.jpg
            print("#################################################")
            print(path_fold)
            cmd = "ffmpeg -i {}  -vf fps=5  {}/%04d.jpg".format(vid_path, path_fold)
            subprocess.call(cmd, shell=True)

