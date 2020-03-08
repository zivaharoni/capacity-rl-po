#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES=0, python ./example.py --exp_name size2  --channel_cardinality 2  --n_clusters 4  --config ./configs/example.json  &
#CUDA_VISIBLE_DEVICES=0, python ./example.py --exp_name size3  --channel_cardinality 3  --n_clusters 6  --config ./configs/example.json  &
#CUDA_VISIBLE_DEVICES=0, python ./example.py --exp_name size4  --channel_cardinality 4  --n_clusters 8  --config ./configs/example.json  &
#CUDA_VISIBLE_DEVICES=0, python ./example.py --exp_name size5  --channel_cardinality 5  --n_clusters 10 --config ./configs/example.json  &
#CUDA_VISIBLE_DEVICES=0, python ./example.py --exp_name size6  --channel_cardinality 6  --n_clusters 12 --config ./configs/example.json  &
#CUDA_VISIBLE_DEVICES=0, python ./example.py --exp_name size7  --channel_cardinality 7  --n_clusters 14 --config ./configs/example.json  &
#CUDA_VISIBLE_DEVICES=0, python ./example.py --exp_name size8  --channel_cardinality 8  --n_clusters 16 --config ./configs/example.json  &
#CUDA_VISIBLE_DEVICES=0, python ./example.py --exp_name size9  --channel_cardinality 9  --n_clusters 20 --config ./configs/example.json  &
#CUDA_VISIBLE_DEVICES=1, python ./example.py --exp_name size10 --channel_cardinality 10 --n_clusters 20 --config ./configs/example.json  &
#CUDA_VISIBLE_DEVICES=1, python ./example.py --exp_name size12 --channel_cardinality 12 --n_clusters 20 --config ./configs/example.json  &
#CUDA_VISIBLE_DEVICES=1, python ./example.py --exp_name size14 --channel_cardinality 14 --n_clusters 20 --config ./configs/example.json  &
#CUDA_VISIBLE_DEVICES=1, python ./example.py --exp_name size16 --channel_cardinality 16 --n_clusters 20 --config ./configs/example.json  &
#CUDA_VISIBLE_DEVICES=2, python ./example.py --exp_name size18 --channel_cardinality 18 --n_clusters 20 --config ./configs/example.json  &
#CUDA_VISIBLE_DEVICES=2, python ./example.py --exp_name size20 --channel_cardinality 20 --n_clusters 20 --config ./configs/example.json  &
#CUDA_VISIBLE_DEVICES=2, python ./example.py --exp_name size25 --channel_cardinality 25 --n_clusters 20 --config ./configs/example.json  &
#CUDA_VISIBLE_DEVICES=3, python ./example.py --exp_name size30 --channel_cardinality 30 --n_clusters 20 --config ./configs/example.json  &
#CUDA_VISIBLE_DEVICES=4, python ./example.py --exp_name size35 --channel_cardinality 35 --n_clusters 20 --config ./configs/example.json  &
#CUDA_VISIBLE_DEVICES=5, python ./example.py --exp_name size40 --channel_cardinality 40 --n_clusters 20 --config ./configs/example.json  &
#CUDA_VISIBLE_DEVICES=6, python ./example.py --exp_name size45 --channel_cardinality 45 --n_clusters 20 --config ./configs/example.json  &
#CUDA_VISIBLE_DEVICES=7, python ./example.py --exp_name size50 --channel_cardinality 50 --n_clusters 20 --config ./configs/example.json  &

CUDA_VISIBLE_DEVICES=2, python ./example.py --exp_name size60-T5  --channel_cardinality 60  --n_clusters 20  --config ./configs/example.json  &
CUDA_VISIBLE_DEVICES=1, python ./example.py --exp_name size70-T5  --channel_cardinality 70  --n_clusters 20  --config ./configs/example.json  &
CUDA_VISIBLE_DEVICES=3, python ./example.py --exp_name size80-T5  --channel_cardinality 80  --n_clusters 20  --config ./configs/example.json  &
#CUDA_VISIBLE_DEVICES=1, python ./example.py --exp_name size90  --channel_cardinality 90  --n_clusters 20  --config ./configs/example.json  &
#CUDA_VISIBLE_DEVICES=3, python ./example.py --exp_name size100 --channel_cardinality 100 --n_clusters 20  --config ./configs/example.json  &
