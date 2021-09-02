# Posepred
Pospred is an open-source toolbox for pose prediction based on PyTorch. It is a part of the VitaLab project.

<br>

<div align="center">
    <img src="visualization/outputs/2D/2D_visualize.gif" width="600px" alt><br>
</div>

## Installation
To get started as quickly as possible, follow the instructions in this section. This should allow you train a model from scratch, evaluate your pretrained models, and produce basic visualizations.

### Dependencies
Make sure you have the following dependencies installed before proceeding:
- Python 3.7+ distribution
- PyTorch >= 1.7.0
- CUDA >= 10.0.0
### Virtualenv
You can create and activate virtual environment like below:
```bash
virtualenv <venvname>

source <venvname>/bin/activate
```
### Requirements
then just install all packages you need:

```bash
pip install -r requirements.txt
```

## Preprocessing
We need to create clean static file to enhance dataloader and speed-up other parts.
To fulfill mentioned purpose You should run preprocessing api called `preprocess.py` like below:

```
usage: python -m api.preprocess [-h] [--dataset_name] [--dataset_path] [--data_usage]
                            [--use_mask] [--interactive] [--output_name]
                            [--obs_frames_num] [--pred_frames_num] [--skip_frame_num]
                            [--use_video_once] [--keypoint_dim]

mandatory arguments:
  --dataset_name        Name of using dataset Ex: 'posetrack' or '3dpw' (str)
  --dataset_path        Path to dataset (str)
  --data_usage          Type of data to use Ex: 'train', 'validation' or 'test' (str)
  --obs_frames_num       Number of frames to observer (int)
  --pred_frames_num      Number of frames to predict (int)
  --skip_num      Number of frame to skip between each two used frames (int)
  --keypoint_dim                 Number of dim data should have Ex: 2 for 2D and 3 for 3D (int)
  --interactive         Create interactive data for interactive models (bool)
  --use_mask            Use visibility mask if possible for dataloader (bool)
  
optional arguments:
  -h, --help            Show this help message and exit
  --use_video_once      Use whole video just once or use until last frame
  --output_name         Name of generated csv file    
```
Example:
```bash
python -m api.preprocess --dataset_name='posetrack' --dataset_path=<path_to_dataset> --data_type='train' --is_interactive --mask
```

## Usage
Here

## Visualization
You can Visualize both 3D and 2D data with visualization module.
In order to generate .gif outputs you can run `visualize.py‍‍‍‍` like below:
```
usage: python -m api.visualize [-h] [--dataset_name] [--keypoint_dim] [--interactive]
                            [--persons_num] [--pred_frames_num] [--skip_frame_num]
                            [--use_mask] [--model_name] [--load_path] [--index] [--ground_truth]

required arguments:
  --dataset_name        Name of using dataset (str)
  --keypoint_dim        Path to dimension of each keypoint (int)
  --persons_num         Number of people in each sequence (int)
  --pred_frames_num      Number of future frames to predict (int)
  --skip_num      Number of frame to skip between each two used frames (int)
  --keypoint_dim                 Number of dim data should have Ex: 2 for 2D and 3 for 3D (int)
  --interactive         To consider interaction or not (bool)
  --use_mask            Use visibility mask if possible for dataloader (bool)
  --model_name          Name of desired model (str)
  --load_path           Path of pretrained model (str)
  --index               Index of a sequence in dataset (int)
  --ground_truth        Use ground-truth future frames (bool)
  --index               Index of a sequence in dataset (int)
  
optional arguments:
  -h, --help            Show this help message and exit
  --load_path           Path of pretrained model (str)
  --ground_truth        Use ground-truth future frames (bool)
      
```

### 2D Visualization

Example:
```bash
python -m api.visualize --dataset_name='sample' --model_name='lstm_vel' --keypoint_dim=2 --load_path='./exps/train/5/snapshots/500.pt' --index=200 --interactive
```

Sample output:

<div align="center">
    <img src="visualization/outputs/2D/2D_visualize.gif" width="600px" alt><br>
</div>

### 3D Visualization

If we have camera extrinsic and intrinsic parameters and image paths, we would create 2 gifs:
- 2D overlay on images
- 3D positions from the camera's POV

Example:
```bash
python -m api.visualize --dataset_name='sample' --model_name='lstm_vel' --keypoint_dim=3--load_path='./exps/train/5/snapshots/500.pt' --index=200 --interactive
```

Sample outputs:
<div align="center">
    <img src="visualization/outputs/2D/3D_visualize_2D_overlay.gif" width="600px" alt><br>
    <img src="visualization/outputs/3D/3D_visualize.gif" width="600px" alt>
</div>


