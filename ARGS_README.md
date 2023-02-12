# Arguments
This is a description to provide details about arguments of Posepred API.
Pospred is an open-source toolbox for pose prediction in PyTorch. It is a part of the VitaLab project.  

# Hydra
```
posepred
├── configs
│   ├── hydra                     
|      ├── data
|         └── main.yaml                 -- main config file for data module (Essentially responsible for creating dataloader)             
|      ├── model
|         ├── pv_lstml.py                     
│         ├── neareset_neighbor.py            
|         ├── zero_vel.py  
|         ├── ...              
|      ├── optimizer
|         ├── adam.yaml                 -- config file for adam optimizer
|         ├── sgd.yaml                  -- config file for stochastic gradient descent optimizer
|         ├── ...   
|      ├── scheduler
|         ├── reduce_lr_on_plateau.yaml -- config file for reducing learning_rate on plateau technique arguments
|         ├── step_lr.yaml              -- config file for step of scheduler arguments                               
|         ├── ...   
|      ├── visualize.yaml               -- config file for visualizer API arguments
|      ├── evaluate.yaml                -- config file for evaluate API arguments 
|      ├── preprocess.yaml              -- config file for preprocess API arguments
|      ├── train.yaml                   -- config file for train API arguments
|      ├── generate_output.yaml         -- config file for generate_output API arguments       
|      └── metrics.yaml                 -- config file for metrics
|                    
└─── logging.conf                 -- logging configurations
```
Now we will precisely explain each module.
#### data
Location: 'configs/hydra/data'

`main.yaml`:
```
Mandatory arguments:
keypoint_dim:               Number of dim data should have Ex: 2 for 2D and 3 for 3D (int)  
is_interactive:             Use interactions between persons (bool) (default: False)
is_noisy:                   Use noisy data_loader `data_loader/noisy_solitary_dataset.py` (bool) (default: False) 
is_pose:             Use pose (dynamic) data_loader (bool) (default: False)
is_solitary:                Use Solitary data_loader (bool) (default: False) 
noise_rate:                 A float number to indicate percentage of uniform noisy values or 'mask' to use mask values as noise (float or str) (default: mask)
persons_num:                Number of people (int)
use_mask:                   Set True To use mask in dataloader and model (bool)
use_quaternion:             Set True To use quaternion representation (based on model) (bool) (default: False)
model_pose_format:          Used data format for pose dataset (str)
metric_pose_format:         Used data format for metrics if pose dataset is used. If no value is specified it'll use the model_pose_format's value (str)
is_h36_testing:             Set True to configure the dataloader for testing huamn3.6m (bool)
is_testing:                 Set True to configure the dataloader for testing (bool) (default: False)
is_visualizing:             Set True to configure dataloader for visualizing (bool) (default: False)
batch_size:                 Indicates size of batch size (int) (default: 256)
shuffle:                    Indicates shuffling the data in dataloader (bool) (default: False)
pin_memory:                 Using pin memory or not in dataloader (bool) (default: False)  
num_workers:                Number of workers (int)
normalize:                  Set True to normalize the data in dataloader (bool)
len_observed:               Number of frames to observe (int)
len_future:                 Number of frames to predict(int)

optional arguments: 
noise_keypoint:             Index of specific keypoint you want to make noisy. (int)
metadata_path:              path to metadata, obligatory when normalize=True
seq_rate:                   The gap between start of two adjacent sequences (1 means no gap) (int) (default: 1) (only used for pose data_loader) 
frame_rate:                 The gap between two frames (1 means no gap) (int) (default: 2) (only used for pose data_loader) 
```
#### model
Folder Location: 'configs/hydra/model'

`common.yaml`:
```
Mandatory arguments:
keypoint_dim:               Number of dim data should have Ex: 2 for 2D and 3 for 3D (int)  
pred_frames_num:            Number of frames to predict, obligatory when ground-truth is not available (int)
obs_frames_num:	            Number of frames to observe (int)
use_mask:                   Set True To use mask in dataloader and model (bool)
use_dct:                    Set True to use discrete cosine transform (bool)
normalize:                  Set True to normalize the data in dataloader (bool)
mean_pose:
std_pose:
device:                     Choose either 'cpu' or 'gpu' (str)
```

`<model-name>.yaml`:

For each model you implement, you should provide a yaml file to configure its argumants.
```
Mandatory arguments:
type:                       Name of the model (str)
loss.type:	            Name of the loss function (str)

optional arguments: 
Every specific argument required for your model!
```

#### optimizer
Folder Location: 'configs/hydra/optimizer'

`adam.yaml`
```
type                        type=adam for adam optimizer (str)
lr                          learning rate (float) (default=0.001)
weight_decay                weight decay coefficient (default: 1e-5)
```
`adamw.yaml`
```
type                        type=adamw for adamw optimizer (str)
lr                          learning rate (float) (default=0.001)
betas                       coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
weight_decay                weight decay coefficient (default: 1e-5)
```
`sam.yaml`
```
type                        type=sam for sharpness aware minimization (str)
lr                          learning rate (float) (default=0.001)
weight_decay                weight decay coefficient (default: 1e-5)
```
`sgd.yaml`
```
type                        type=sgd for stochastic gradient descent (str)
lr                          learning rate (float) (default=0.001)
momentum                    momentum factor in sgd optimizer (float) (default=0)
dampening                   dampening for momentum in sgd optimizer (float) (default=0)
weight_decay                weight decay coefficient (default: 1e-5)
nesterov                    enables Nesterov momentum (bool) (default=False)
```

#### scheduler
Folder Location: 'configs/hydra/scheduler'

`multi_step_lr.yaml`
```
type                        type=multi_step_lr to use this technique
step_size                   List of epoch indices. Must be increasing.
gamma                       Multiplicative factor of learning rate decay. (float) (default=0.4)
```
`reduce_lr_on_plateau.yaml`
```
type                        type=reduce_lr_on_plateau to use this technique (str)
mode                        One of `min`, `max`. In `min` mode, lr will be reduced when the quantity monitored has stopped
                            decreasing; in `max` mode it will be reduced when the quantity monitored has stopped increasing (str) (default=min)     
factor                      actor by which the learning rate will be reduced. new_lr = lr * factor (float) (default=0.5)
patience                    Number of epochs with no improvement after which learning rate will be reduced. (int) (default=20)
threshold                   Threshold for measuring the new optimum, to only focus on significant changes (float) (default=le-3)
verbose                     If True, prints a message to stdout for each update. (bool) (defaulTrue)
```
`step_lr.yaml`
```
type                        type=step_lr to use this technique
step_size                   Period of learning rate decay (int) (default=50)
gamma                       Multiplicative factor of learning rate decay. (float) (default=0.5)
last_epoch                  The index of last epoch (int) (default=-1)
verbose                     If True, prints a message to stdout for each update (bool) (default=False)
```
`no_step_lr.yaml`
```
type                        type=no_step_lr to use this technique
step_size                   Period of learning rate decay (int) (default=50)
gamma                       Multiplicative factor of learning rate decay. (float) (default=0.5)
last_epoch                  The index of last epoch (int) (default=-1)
verbose                     If True, prints a message to stdout for each update (bool) (default=False)
```
#### metrics
File Location: 'configs/hydra/metrics.yaml'

`metrics.yaml`:
```
pose_metrics:               List which metrics in the metrics module you want to use.
mask_metrics:               List which metrics in the metrics module you want to use.
```


## Preprocessing   
Check preprocessing config file: "configs/hydra/preprocess.yaml" for more details.

You can change preprocessor via commandline like below:
```  
usage: python -m api.preprocess  	[official_annotation_path] [dataset] [data_type]                          	
	                                [obs_frames_num] [pred_frames_num] [keypoint_dim]
				                    [use_video_once] [skip_num] [interactive]  
	                             	[annotate_openpifpaf] [annotate] [image_dir]
	                             	[output_name] [save_total_frames]
  
mandatory arguments:
  - official_annotation_path  Name of using dataset Ex: 'posetrack' or '3dpw' (str)  
  - dataset             Name of Dataset []  
  - data_type           Type of data to use Ex: 'train', 'validation' or 'test' (str)  
  - obs_frames_num      Number of frames to observe (int)  
  - pred_frames_num     Number of frames to predict (int)    
  - keypoint_dim        Number of dim data should have Ex: 2 for 2D and 3 for 3D (int) 
    
optional arguments:  
  - interactive         Use interactions between persons (bool)
  - annotate		(only for JAAD and PIE) Implies that to use dataset_path as annotation path or a path to images. in this context, we generate annotations using openpifpaf then create static files afterward. (bool)
  - image_dir           (only for JAAd and PIE) define this parameter if annotate = True and it implies the directory for images to creat annotation based on. (str)
  - annotation_path     (only for JAAD and PIE) define this parameter if annotate = False and it indicates the path to annotations. (str)  
  - skip_num        	Number of frame to skip between each two used frames (int) (0 implies no skip) (str)
  - load_60Hz           This value is used only for 3DPW
  - use_video_once      Use whole video just once or use until last frame (bool) (default: false)  
  - output_name         Name of generated csv file (str) (for default we have specific conventions for each dataset)
  - save_total_frames   This value must be 'true' for pose_dataset and flase for other datasets (bool) (default: false)
  
```  
Example:  
```bash
python -m api.preprocess \
    dataset=human3.6m \
    official_annotation_path=$DATASET_PATH \
    data_type=train \
    keypoint_dim=3 \
    interactive=false \
    output_name=new_full \
    save_total_frames=true \
    obs_frames_num=10 \
    pred_frames_num=25
```
  
## Training
Check training config file: "configs/hydra/train.yaml" for more details.

You can change training args via command line like below:
```  
usage: python -m api.train      [data] [model] [optimizer] [scheduler]
				            [train_dataset] [valid_dataset] [keypoint_dim] 
                           	[epochs] [start_epoch] [device]
                           	[use_mask] [use_dct] [normalize] [is_noisy]
				            [snapshot_interval] [load_path] [save_dir]
                            [obs_frames_num] [pred_frames_num]
                            [model_pose_format] [metric_pose_format]
                            [experiment_name] [mlflow_tracking_uri]

mandatory arguments:
  data                  Name of the dataloader yaml file, default is main dataloader (str)
  model                 Name of the model yaml file (str)
  optimizer             Name of the optimizer yaml file, default is adam (str)
  scheduler             Name of the scheduler yaml file, default is reduce_lr_on_plateau (str)
  train_dataset         Path of the train dataset (str)   
  keypoint_dim          Dimension of the data Ex: 2 for 2D and 3 for 3D (int)  
  epochs                Number of training epochs (int)

optional arguments:
  - valid_dataset       Path of validation dataset (str)    
  - use_mask            Consider visibility mask (bool)
  - use_dct             Consider using dct (bool)
  - normalize           Normalize the data or not (bool)
  - snapshot_interval 	Save snapshot every N epochs (int)
  - load_path           Path to load a model (str) 
  - start_epoch	 	Start epoch (int)
  - device              Choose either 'cpu' or 'cuda' (str)
  - save_dir            Path to save the model (str)
  - obs_frames_num      Number of observed frames for pose dataset (int) (default: 10)
  - pred_frames_num     Number of future frames for pose dataset (int) (default:25)
  - model_pose_format   Used data format for pose dataset (str) (defautl: total -> for more information see the Data section)
  - metric_pose_format  Used data format for metrics if pose dataset is used. If no value is specified it'll use the model_pose_format's value
  - experiment_name:    Experiment name for MLFlow (str) (default: "defautl experiment")
  - mlflow_tracking_uri:  Path for mlruns folder for MLFlow (str) (default: saves mlruns in the current folder)

```  

Example:
```bash  
python -m api.train model=history_repeats_itself \
          keypoint_dim=3 \
          train_dataset=$DATASET_TRAIN_PATH \
          valid_dataset=$DATASET_TEST_PATH \
          epochs=10 \
          data.shuffle=True device=cuda \
          snapshot_interval=1 \
          hydra.run.dir=$OUTPUT_PATH \
          data.is_pose=True \
          data.batch_size=256 \
          data.num_workers=10 \
          data.seq_rate=2 \
          obs_frames_num=50 \
          pred_frames_num=25 \
          model_pose_format=xyz \
          metric_pose_format=xyz \
          optimizer=adam \
          optimizer.lr=0.007 \
          experiment_name=his_encoder
```  

## Evaluation
Check evaluation config file: "configs/hydra/evaluate.yaml" for more details.

You can change evaluation args via command line like below:
``` 
usage: python -m api.evaluate      [data] [model] [dataset] [keypoint_dim] 
                              	   [use_mask] [normalize] [is_noisy]
                              	   [device] [rounds_num] [load_path] 
                                   [obs_frames_num] [pred_frames_num] 
                                   [model_pose_format] [metric_pose_format]

mandatory arguments:
  data          Name of the dataloader yaml file, default is main dataloader (str)
  model         Name of the model yaml file (str)
  dataset       Name of dataset Ex: 'posetrack' or '3dpw' (str)    
  keypoint_dim  Number of dim data should have Ex: 2 for 2D and 3 for 3D (int)  
  load_path     Path to load a model (str)
						   
optional arguments:
  - use_mask    Consider visibility mask (bool)
  - normalize   Normalize the data or not (bool)
  - is_noisy    Whether data is noisy or not (bool)
  - device      Choose either 'cpu' or 'gpu' (str)
  - obs_frames_num      Number of observed frames for pose dataset (int) (default: 10)
  - pred_frames_num     Number of future frames for pose dataset (int) (default:25)
  - model_pose_format   Used data format for pose dataset (str) (defautl: total -> for more information see the Data section)
  - metric_pose_format  Used data format for metrics if pose dataset is used. If no value is specified it'll use the model_pose_format's value
```  

Example:
```bash  
python -m api.evaluate model=msr_gcn \
          keypoint_dim=3 \
          dataset=$DATASET_TEST_PATH \
          data.shuffle=True \
          rounds_num=1 \
          device=cuda \
          hydra.run.dir=$OUTPUT_PATH \
          data.is_pose=True \
          data.batch_size=2048 \
          obs_frames_num=10 \
          pred_frames_num=25 \
          model_pose_format=xyz \
          metric_pose_format=xyz \
          load_path=$MODEL_PATH
```  
another example:
```bash
python -m api.evaluate model=zero_vel \
          keypoint_dim=3 \
          dataset=$DATASET_TEST_PATH \
          data.shuffle=True \
          rounds_num=1 \
          device=cuda \
          hydra.run.dir=$OUTPUT_PATH \
          data.is_pose=True \
          data.batch_size=2048 \
          obs_frames_num=10 \
          pred_frames_num=25 \
          model_pose_format=xyz \
          metric_pose_format=xyz 
```

## Generating Outputs

```  
usage: python -m api.generate_final_output	[data] [model] [dataset] 
						[pred_frames_num] [keypoint_dim] [load_path]
                             	  		[use_mask] [normalize] [is_noisy]
                             	  		[device] [save_dir]
                             	  								 
mandatory arguments:
  data			Name of the dataloader yaml file, default is main dataloader (str)
  model			Name of the model yaml file (str)
  dataset    		Name of dataset Ex: 'posetrack' or '3dpw' (str)    
  keypoint_dim          Number of dim data should have Ex: 2 for 2D and 3 for 3D (int)  
  load_path  		Path to load a model (str)  
  pred_frames_num 	Number of frames to predict. Mandatory if load_path is None. (int)
						   
optional arguments:
  use_mask 		Consider visibility mask (bool)
  normalize		Normalize the data or not (bool)
  save_dir              Path to save the model (str)
  is_noisy		Whether data is noisy or not (bool)
  device		Choose either 'cpu' or 'gpu' (str)
```  

Example:
```bash  
python -m api.predict dataset=<dataset_name> keypoint_dim=2 load_path=<path_to_model> 
```  
```bash  
python -m api.predict dataset=<dataset_name> model=<model_name> keypoint_dim=2 pred_frames_num=<pred_frames_num>
```  
    
## Visualization  
You can directly change config file: "congifs/hydra/visualize.yaml". Note that you need your model and data: "configs/hydra/data/main.yaml" configs but the default ones should be fine.

Also, all essential changes you need are defined below:
```  
usage: python -m api.visualize [dataset_type] [dataset] [model]
                               [images_dir] [save_dir] [index] [normalize] [is_noisy] [use_mask]
                               [pred_frames_num] [load_path] [device] [skip_num]
                               [data.is_visualizing]  
                               [data.X for all arguments in configs/hydra/data]
                               
mandatory arguments:  
    dataset_type 	    Name of using dataset. (str) (['somof_posetrack', 'posetrack', 'somof_3dpw', '3dpw', 'jta', 'jaad', 'pie', 'human3.6m'])  
    model        	    Name of desired model. (str) (['zero_vel','nearest_neighbor', 'pv_lstm', 'disentangled', 'derpof', 'history_repeats_itself', 'mix_and_match', 'comp_pred_vel', 'comp_pred_pose','comp_pred_center', 'comp_pred_root','trans_cvae','pv_lstm_noisy', 'comp_pred_vel_concat', 'v_lstm_noisy', 'p_lstm_noisy', 'pv_lstm_pro', 'keyplast'])  
    images_dir 		    Path to existing images on your local computer (str)
    showing                 Indicates which images we want to show (dash(-) separated list) ([observed, future, predicted, completed])   
    index                   Index of a sequence in dataset to visualize. (int)
    normalize               Normalize the data or not (bool)
    data.normalize          If data.normalize = True, we have to pass <data.metadata_path>. (bool)
    data.metadata_path      Path to metadata of dataset. (str)
    data.is_visualizing     Essentially indicates if we are using visualizer, MUST be True for this part (bool) (default: True)
    
    
optional arguments:  
  load_path             Path to pretrained model. Mandatory if using a training-based model (str)  
  pred_frames_num 	Number of frames to predict. Mandatory if load_path is None. (int)
  interactive           To consider interaction or not. (bool)  
  use_mask              Use visibility mask if possible for dataloader. (bool)  
  skip_num     	        Number of frame to skip between each two used frames. (int)
```  
  
### 2D Visualization  
  
Example:  
```bash  
python -m api.visualize --dataset=<dataset_name> --model=<model_name> --keypoint_dim=2 --load_path=<path_to_model> --images_dir=<images_dir>
```  

### 3D Visualization  

```bash  
python -m api.visualize --dataset=<dataset_name> --model=<model_name> --keypoint_dim=3 --load_path=<path_to_model> --images_dir=<images_dir>
```  
