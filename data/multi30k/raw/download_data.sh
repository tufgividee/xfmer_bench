#!/bin/bash

set -euo pipefail

wget https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/train.en.gz
wget https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/train.de.gz
wget https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/val.en.gz
wget https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/val.de.gz
wget https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2016_flickr.en.gz
wget https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2016_flickr.de.gz

gunzip -vf *.gz