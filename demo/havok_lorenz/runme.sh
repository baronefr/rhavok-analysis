#!/bin/bash

#  This script runs the realtime demo of Lorenz attractor.
#  
#  OS requirements:  Ubuntu
# 
#  >> 09 may 2022, Francesco Barone
#  Laboratory of Computational Physics, University of Padua

fifoname='data.pipe'

mkfifo $fifoname

gnome-terminal -- $(pwd)/generator.py ${fifoname} &
./elaborator.py ${fifoname}

rm $fifoname
echo '--end of demo--'
