#! /bin/bash

ffmpeg -r 12 -f image2 -i %d.png -vcodec gif -y wfc.gif
