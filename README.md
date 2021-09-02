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
	                            [--obs_frames_num] [--pred_frames_num] [--keypoint_dim]
                				[--interactive] [--use_mask] [--skip_num]  
	                            [--use_video_once] [--output_name]
  
mandatory arguments:  
  --dataset_name        Name of using dataset Ex: 'posetrack' or '3dpw' (str)  
  --dataset_path        Path to dataset (str)  
  --data_usage          Type of data to use Ex: 'train', 'validation' or 'test' (str)  
  --obs_frames_num      Number of frames to observe (int)  
  --pred_frames_num     Number of frames to predict (int)    
  --keypoint_dim        Number of dim data should have Ex: 2 for 2D and 3 for 3D (int)  
    
optional arguments:  
  -h, --help            Show this help message and exit  
  --interactive         Use interactions between persons (bool)
  --use_mask            Use visibility mask for dataloader (bool)
  --skip_num      		Number of frame to skip between each two used frames (int)
  --use_video_once      Use whole video just once or use until last frame (bool)  
  --output_name         Name of generated csv file (str)
```  
Example:  
```bash  
python -m api.preprocess --dataset_name='posetrack' --dataset_path=<path_to_dataset> --data_usage='train' --obs_frames_num=16 --pred_frames_num=14 --use_mask  
```  
  
## Training
Simple training script:
```bash  
python -m api.train --train_dataset=<dataset_name> --model=<model_name> --keypoint_dim=2 --epochs=100
```  
Provide **validation_dataset** to adjust learning-rate and report metrics on validation-dataset as well.
See here for more details about training arguments.


## Evaluation
Evaluate pretrained model:
```bash  
python -m api.evaluate --dataset=<dataset_name> --model=<model_name> --keypoint_dim=2 --load_path=<path_to_model> 
```  
See here for more details about evaluation arguments.


## Prediction
Generate and save predicted future pose:
```bash  
python -m api.predict --dataset=<dataset_name> --model=<model_name> --keypoint_dim=2 --load_path=<path_to_model> 
```  
See here for more details about prediction arguments.
  
  
## Visualization  
You can Visualize both 3D and 2D data with visualization module.  
In order to generate .gif outputs you can run `visualize.py‍‍‍‍` like below:  
```  
usage: python -m api.visualize [-h] [--dataset] [--model] [--keypoint_dim]
							   [--persons_num] [--index] [--load_path] [--ground_truth]
							   [--pred_frames_num] [--interactive] [--use_mask][--skip_num]  
  
mandatory arguments:  
  --dataset 	        Name of using dataset (str)  
  --model          Name of desired model (str)  
  --keypoint_dim        Number of dim data should have Ex: 2 for 2D and 3 for 3D (int)  
  --persons_num         Number of people in each sequence (int)
    
optional arguments:  
  -h, --help            Show this help message and exit  
  --index               Index of a sequence in dataset to visualize (int)  
  --load_path           Path to pretrained model (str)  
  --ground_truth        Visualize ground-truth future frames as well (bool)  
  --pred_frames_num     Number of future frames to predict, mandatory when not using GT (int)  
  --interactive         To consider interaction or not (bool)  
  --use_mask            Use visibility mask if possible for dataloader (bool)  
  --skip_num      		Number of frame to skip between each two used frames (int)
        
```  
  
### 2D Visualization  
  
Example:  
```bash  
python -m api.visualize --dataset='sample' --model='lstm_vel' --keypoint_dim=2 --load_path=<path_to_model>
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
python -m api.visualize --dataset='sample' --model='lstm_vel' --keypoint_dim=3 --load_path=<path_to_model>
```  
  
Sample outputs:  
<div align="center">  
    <img src="visualization/outputs/2D/3D_visualize_2D_overlay.gif" width="600px" alt><br>  
    <img src="visualization/outputs/3D/3D_visualize.gif" width="600px" alt>  
</div>