#!/bin/bash

data_dir="/home/hmb1604/datasets/LivDet/2015"
out_dir="../data/2015"
sensors=("CrossMatch" "DigitalPersona" "GreenBit" "HiScan")
block_sizes=(3 3 3 3)
deltas=(2 2 36 18)
kernel_sizes=(15 3 19 19)

for i in "${!sensors[@]}"; do
    python foreground_extraction.py -i "${data_dir}/${sensors[i]}" -o "$out_dir/${sensors[i]}" -b ${block_sizes[i]} -d ${deltas[i]} -k ${kernel_sizes[i]}
done

# i=2
# python foreground_extraction.py -i "${data_dir}/${sensors[$i]}" -o "$out_dir/${sensors[$i]}" -b ${block_sizes[i]} -d ${deltas[i]} -k ${kernel_sizes[i]}