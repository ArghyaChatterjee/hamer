#!/bin/bash

# Define variables
GDRIVE_ID="1mv7CUAnm73oKsEEG1xE3xH2C_oqcFSzT"
TAR_FILE="hamer_demo_data.tar.gz"

# Alternatively, you can use wget
#wget https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz

# Download using gdown
echo "Downloading model from Google Drive..."
gdown "https://drive.google.com/uc?id=$GDRIVE_ID"

# Extract the tar file
if [ -f "$TAR_FILE" ]; then
    echo "Extracting $TAR_FILE..."
    tar --warning=no-unknown-keyword --exclude=".*" -xvf "$TAR_FILE"

    # Delete the original tar file
    echo "Cleaning up..."
    rm "$TAR_FILE"
else
    echo "Download failed or file not found!"
fi

echo "Done!"


