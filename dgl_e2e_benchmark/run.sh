#!/bin/bash
cd ./fastgcn
./run.sh > outputs/log_new
cd ../ladies
./run.sh > outputs/log_new
cd ../pinsage
./run.sh > outputs/log_new