#!/usr/bin/bash

offset_values=(-3 -2.5 -2)
scale_values=(15 20 25 30)

# String comparison in Bash using [[ ]]
if [[ "$(hostname)" == "ssuhai_GPU1-dj_gpu4" ]]; then


  # Case 2: scale_by_bc_number = true
  for offset in "${offset_values[@]}"; do
    for scale in "${scale_values[@]}"; do
      python train.py output_offset="${offset}" output_scale="${scale}" scale_by_bc_number=true
    done
  done

else
  echo "Not on specified host! Will not run scripts to avoid using the head node."
fi