#!/bin/bash

wget https://doc-0k-5k-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/leop05q9t3406jh53jvkjsatse027o71/1545386400000/05918670026020017381/*/16D-Ob5qTmuVBwc7nGmeE4jallRaqcKkw?e=download -O data/backwards.zip

# unzip
cd data
unzip backwards.zip -d data
rm -f backwards.zip

# move files
mv backwards/driving_log.csv driving_log_backwards.csv
find IMG/ -type f | wc -l # print number of images before
mv backwards/IMG/* IMG/
find IMG/ -type f | wc -l # print number of images after

rm -rf backwards

cd ..
