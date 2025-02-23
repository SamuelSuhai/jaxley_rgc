#!/usr/bin/bash

batchsizes=(16 8 4 2 1)

# String comparison in Bash using [[ ]]
if [[ "$(hostname)" == "ssuhai_GPU1-dj_gpu4" ]]; then


  # loop over batch sizes
  for bs in "${batchsizes[@]}"; do
      echo "Training with batch size ${bs}..."
      if python train.py batchsize="${bs}"; then
          echo "Training with batch size ${bs} successful. Exiting loop."
          break  # Exit the loop if successful
      else
          echo "Training with batch size ${bs} failed. Continuing to the next batch size."
      fi
  done

else
  echo "Not on specified host! Will not run scripts to avoid using the head node."
fi