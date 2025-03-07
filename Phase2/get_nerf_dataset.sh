#!/bin/bash
curl -L -o nerf-synthetic-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/nguyenhung1903/nerf-synthetic-dataset
unzip nerf-synthetic-dataset.zip
rm nerf-synthetic-dataset.zip
