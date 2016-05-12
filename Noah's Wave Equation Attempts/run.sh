#!/usr/local/bin/bash
# runs all py files in a given directory

for item in $1/*.py
do
    python $1/$item
    mkdir $1/videos
    mv $1/*.mp4 $1/videos
done