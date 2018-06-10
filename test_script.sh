#!/bin/sh
if [ "$#" -ne 1 ]; then
  echo "Need the name of the data directory as argument (Single argument only)"
  exit 1
fi

echo "Chosen dataset: $1"
python experiment.py --d $1 -e 10 -nh 1
python experiment.py --d $1 -e 100 -nh 1
python experiment.py --d $1 -e 500 -nh 2
