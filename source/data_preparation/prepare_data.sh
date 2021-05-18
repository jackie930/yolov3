#!/bin/bash
# Download labels, change this path
mkdir data
mkdir data/Annotations
aws s3 cp s3://open-source-models/esd_data/labels/ ./data/Annotations --recursive
# download images, change this path
mkdir data/images
aws s3 cp s3://open-source-models/esd_data/images/ ./data/images --recursive

#process the files
python voc_label.py

#move files
mkdir ../data
rm -r ../data/custom
mkdir ../data/custom
mv data/ ../data/custom/


