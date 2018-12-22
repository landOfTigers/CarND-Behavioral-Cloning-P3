#!/bin/bash

cd data

# download
wget https://doc-0k-5k-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/gfc3f4eg8njci1bchgv62lq8baa946m2/1545472800000/05918670026020017381/*/16D-Ob5qTmuVBwc7nGmeE4jallRaqcKkw?e=download -O backwards.zip
wget https://doc-0o-5k-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/6b5rfef9mut57cd3n1i1ok18ehqpi8gn/1545472800000/05918670026020017381/*/11WtTIN3Oe4GDAoevAaj6G6AFvBFFIfPH?e=download -O recovery.zip

# unzip
unzip backwards.zip -d .
rm -f backwards.zip
unzip recovery.zip -d .
rm -f recovery.zip

# move files
mv backwards/driving_log.csv driving_log_backwards.csv
find IMG/ -type f | wc -l # print number of images before
mv backwards/IMG/* IMG/
find IMG/ -type f | wc -l # print number of images after
rm -rf backwards
mv recovery2/driving_log.csv driving_log_recovery.csv
find IMG/ -type f | wc -l # print number of images before
mv recovery2/IMG/* IMG/
find IMG/ -type f | wc -l # print number of images after
rm -rf recovery2

cd ..
