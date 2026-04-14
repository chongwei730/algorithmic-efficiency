#!/bin/bash
#SBATCH -J download_data
#SBATCH -p agsmall         
#SBATCH -t 24:00:00            
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH -o download.%j.out
#SBATCH -e download.%j.err

source ~/.bashrc
conda activate ae

# python datasets/dataset_setup.py --librispeech \
#   --data_dir=/scratch.global/chen8596/mlcommons_data \
#   --temp_dir=/scratch.global/chen8596/tmp/mlcommons \
#   --skip_download

# python datasets/dataset_setup.py --fastmri \
#   --data_dir=/scratch.global/chen8596/mlcommons_data \
#   --temp_dir=/scratch.global/chen8596/tmp/mlcommons \
#   --fastmri_knee_singlecoil_train_url "https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_singlecoil_train.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=MKcHQcHolhMNdoKpAyvjitpcuus%3D&Expires=1774564256" \
#   --fastmri_knee_singlecoil_val_url   'https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_singlecoil_val.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=gE%2BME3Wuyuw0OZaaT1Gxxidj%2FfQ%3D&Expires=1774564256' \
#   --fastmri_knee_singlecoil_test_url  'https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_singlecoil_test.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=FADD0%2F6vMs8Xxmu9kfbzSYabgKw%3D&Expires=1774564256'


python datasets/dataset_setup.py --wmt \
  --data_dir=/scratch.global/chen8596/mlcommons_data \
  --temp_dir=/scratch.global/chen8596/tmp/mlcommons \


# python datasets/dataset_setup.py --criteo1tb \
#   --data_dir=/scratch.global/chen8596/mlcommons_data/criteo1tb \
#   --temp_dir=/scratch.global/chen8596/mlcommons_data/temp \
#   --skip_download