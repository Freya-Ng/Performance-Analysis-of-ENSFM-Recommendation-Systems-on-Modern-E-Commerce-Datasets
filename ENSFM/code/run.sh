#!/bin/bash

echo "Running dataset: Arts & Photography"
python3 ENSFM.py --dataset "Arts & Photography"

echo "Running dataset: Genre Fiction"
python3 ENSFM.py --dataset "Genre Fiction"

echo "Running dataset: History"
python3 ENSFM.py --dataset "History"

echo "Finished training all 3 datasets."

# Pause at the end
read -p "Press Enter to exit..."
