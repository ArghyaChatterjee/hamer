# HaMeR: Hand Mesh Recovery

## Code workflow

The whole workflow runs the following way:

- Runs human detection using ViTDet.
- Extracts hand keypoints using ViTPose.
- Computes bounding boxes for the hands.
- Runs the HaMeR model to generate 3D hand meshes.
- Renders the meshes and overlays them on the webcam feed or offline images.
- Displays the processed video stream in real-time or offline.

## Requirements
This has been tested on Ubuntu 22.4 machine with Nvidia 4070 RTX GPU and cuda 12.1. 

## Installation
Clone the repo:
```
git clone https://github.com/ArghyaChatterjee/hamer.git
cd hamer
```

We recommend creating a virtual environment for HaMeR. You can use venv:
```bash
python3.10 -m venv hamer_venv
source hamer_venv/bin/activate
pip3 install --upgrade pip setuptools wheel
```

Then, you can install the rest of the dependencies. This is for CUDA 12.1, but you can adapt accordingly:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip3 install -e .[all]
pip3 install -v -e third-party/ViTPose
```

If you want to install apex:
```bash
cd hamer
git clone https://github.com/NVIDIA/apex
cd apex
pip3 install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

You also need to download the trained models:
```bash
bash fetch_demo_data.sh
```

Besides these files, you also need to download the MANO model. Please visit the [MANO website](https://mano.is.tue.mpg.de) and register to get access to the downloads section.  We only require the right hand model. You need to put `MANO_RIGHT.pkl` under the `_DATA/data/mano` folder.

## Offline Demo

If you want an offline demo, use this script:
```bash
python3 offline_demo.py --img_folder example_data --out_folder demo_out --batch_size=48 --side_view --save_mesh --full_frame --body_detector regnety
```
We are using `regnety` as the human body detector in stead of `vitdet`. If you want to use the `vitdet`, change the `--body_detector` to `regnety`. If your code doesn't run, then change the batch size to `32`, `16`, `8`, `4`, `2` and `1`. 

## Online Demo

If you want an online demo, use this script:
```bash
python3 online_demo.py 
```
In this script, `batch_size=48` and `body_detector` is set to `regnety`. Here is how the workflow starts:

1️⃣ Captures frames from the webcam using OpenCV.

2️⃣ Runs human detection using ViTDet.

3️⃣ Extracts hand keypoints using ViTPose.

4️⃣ Computes bounding boxes for the hands.

5️⃣ Runs the HaMeR model to generate 3D hand meshes.

6️⃣ Renders the meshes and overlays them on the webcam feed.

7️⃣ Displays the processed video stream in real-time.

8️⃣ Press 'Q' to exit.


## Visualize the Hand Mesh model

In order to visualize the hand model, run this script:
```bash
python3 visualize_hand_model_pickle.py
```

<p align="center">
  <img src="assets/mano_hand_mesh.png" alt="Hand Mesh Model" width="500">
</p>

If you want to convert the hand mesh model from `.pkl` to an `.obj` file, run the following script:
```bash
python3 export_hand_model_from_pkl_to_obj.py
```
The input for this script is `MANO_RIGHT.pkl`. Once you convert, you can openup the `MANO_RIGHT.obj` file in blender or meshlab. 

## HInt Dataset
Annotations for the HInt dataset has been released. Please follow the instructions [here](https://github.com/ddshan/hint)

## Training
First, download the training data to `./hamer_training_data/` by running:
```
bash fetch_training_data.sh
```

Then you can start training using the following command:
```bash
python3 train.py exp_name=hamer data=mix_all experiment=hamer_vit_transformer trainer=gpu launcher=local
```
Checkpoints and logs will be saved to `./logs/`.

## Evaluation
Download the [evaluation metadata](https://www.dropbox.com/scl/fi/7ip2vnnu355e2kqbyn1bc/hamer_evaluation_data.tar.gz?rlkey=nb4x10uc8mj2qlfq934t5mdlh) to `./hamer_evaluation_data/`. Additionally, download the FreiHAND, HO-3D, and HInt dataset images and update the corresponding paths in  `hamer/configs/datasets_eval.yaml`.

Run evaluation on multiple datasets as follows, results are stored in `results/eval_regression.csv`. 
```bash
python eval.py --dataset 'FREIHAND-VAL,HO3D-VAL,NEWDAYS-TEST-ALL,NEWDAYS-TEST-VIS,NEWDAYS-TEST-OCC,EPICK-TEST-ALL,EPICK-TEST-VIS,EPICK-TEST-OCC,EGO4D-TEST-ALL,EGO4D-TEST-VIS,EGO4D-TEST-OCC'
```

Results for HInt are stored in `results/eval_regression.csv`. For [FreiHAND](https://github.com/lmb-freiburg/freihand) and [HO-3D](https://codalab.lisn.upsaclay.fr/competitions/4318) you get as output a `.json` file that can be used for evaluation using their corresponding evaluation processes.

## Work with Custom Dataset
You will find the HInt dataset annotations [here](https://github.com/ddshan/hint).

## Install using Docker 

If you wish to use HaMeR with Docker, you can use the following command:

```
docker compose -f ./docker/docker-compose.yml up -d
```

After the image is built successfully, enter the container and run the steps as above:

```
docker compose -f ./docker/docker-compose.yml exec hamer-dev /bin/bash
```

Continue with the installation steps:

```bash
bash fetch_demo_data.sh
```

