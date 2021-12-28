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
noise_rate:                 A float number to indicate percentage of uniform noisy values or 'mask' to use mask values as noise (float or str) (default: mask)
use_mask:                   Set True To use mask in dataloader and model (bool)
use_quaternion:             Set True To use quaternion representation (based on model) (bool) (default: False)
is_testing:                 Set True to configure the dataloader for testing (bool) (default: False)
is_visualizing:             Set True to configure dataloader for visualizing (bool) (default: False)
batch_size:                 Indicates size of batch size (int) (default: 256)
shuffle:                    Indicates shuffling the data in dataloader (bool) (default: False)
pin_memory:                 Using pin memory or not in dataloader (bool) (default: False)  
num_workers:                Number of workers (int)
normalize:                  Set True to normalize the data in dataloader (bool)

optional arguments: 
overfit:                    Set True to create a dataloader with small size for testing overfitting (bool)
noise_keypoint:             Index of specific keypoint you want to make noisy. (int)
metadata_path:              path to metadata, obligatory when normalize=True
```
#### model
Folder Location: 'configs/hydra/model'

`common.yaml`:
```
Mandatory arguments:
keypoint_dim:               Number of dim data should have Ex: 2 for 2D and 3 for 3D (int)  
pred_frames_num:	    Number of frames to predict, obligatory when ground-truth is not available (int)
use_mask:                   Set True To use mask in dataloader and model (bool)
use_dct:                    Set True to use discrete cosine transform (bool)
normalize:                  Set True to normalize the data in dataloader (bool)
```


#### optimizer
Folder Location: 'configs/hydra/optimizer'

`adam.yaml`
```
type                        type=adam for adam optimizer (str)
lr                          learning rate (float) (default=0.001)
```
`sgd.yaml`
```
type                        type=sgd for stochastic gradient descent (str)
lr                          learning rate (float) (default=0.001)
momentum                    momentum factor in sgd optimizer (float) (default=0)
dampening                   dampening for momentum in sgd optimizer (float) (default=0)
nesterov                    enables Nesterov momentum (bool) (default=False)
```

#### scheduler
Folder Location: 'configs/hydra/scheduler'

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
#### metrics
File Location: 'configs/hydra/metrics.yaml'

#### evaluate
File Location: 'configs/hydra/evaluate.yaml'
#### train
File Location: 'configs/hydra/train.yaml'
#### visualize
File Location: 'configs/hydra/visualize.yaml
#### preprocess
File Location: 'configs/hydra/preprocess.yaml
#### generate_output
File Location: 'configs/hydra/generate_output.yaml
## Preprocessing   
Check preprocessing config file: "configs/hydra/preprocess.yaml" for more details.

You can change preprocessor via commandline like below:
```  
usage: python -m api.preprocess  [official_annotation_path] [dataset] [data_type]                          	
	                             [obs_frames_num] [pred_frames_num] [keypoint_dim]
				                 [use_video_once] [skip_num] [interactive]  
	                             [annotate_openpifpaf] [annotate] [image_dir]
	                             [output_name]
  
mandatory arguments:  
  official_annotation_path        Name of using dataset Ex: 'posetrack' or '3dpw' (str)  
  dataset             Name of Dataset []  
  data_type          Type of data to use Ex: 'train', 'validation' or 'test' (str)  
  obs_frames_num      Number of frames to observe (int)  
  pred_frames_num     Number of frames to predict (int)    
  keypoint_dim        Number of dim data should have Ex: 2 for 2D and 3 for 3D (int)  
    
optional arguments:  
  interactive         Use interactions between persons (bool)
  annotate		    (only for JAAD and PIE) Implies that to use dataset_path as annotation path or a path to images. in this context, we generate annotations using openpifpaf then create static files afterward. (bool)
  image_dir           (only for JAAd and PIE) define this parameter if annotate = True and it implies the directory for images to creat annotation based on. (str)
  annotation_path     (only for JAAD and PIE) define this parameter if annotate = False and it indicates the path to annotations. (str)  
  skip_num        	Number of frame to skip between each two used frames (int) (0 implies no skip) (str)
  use_video_once      Use whole video just once or use until last frame (bool) (default: false)  
  output_name         Name of generated csv file (str) (for default we have specific conventions for each dataset)
  
```  
Example:  
```bash
python3 -m api.preprocess dataset=<dataset_name> official_annotation_path=<path_to_dataset> skip_num=<skip_number> obs_frames_num=<number_of_observed_frames> pred_frames_num=<number_of_predicted_frames> keypoint_dim=<number_of_keypoints> data_type=<data_type> interactive=<interactive_true_or_false>
```
  
## Training

```  
usage: python -m api.train [-h] [--train_dataset] [--valid_dataset] [--model] 
                           	[--keypoint_dim] [--epochs] [--start_epoch] 
                           	[--use_mask] [--interactive] [--persons_num]
                           	[--distance_loss] [--mask_loss_weight] [--skip_num]
                           	[--lr] [--decay_factor] [--decay_patience]
                           	[--snapshot_interval] [--load_path]
                           	[--batch_size] [--num_workers] [--shuffle] [--pin_memory]
                           	[--hidden_size] [--n_layers] [--hardtanh_limit] 
                           	[--dropout_enc] [--dropout_pose_dec] [--dropout_mask_dec]

mandatory arguments:
  --train_dataset       Name of train dataset Ex: 'posetrack' or '3dpw' (str)  
  --valid_dataset       Name of validation dataset Ex: 'posetrack' or '3dpw' (str)  
  --model            	Name of desired model (str)  
  --keypoint_dim        Number of dim data should have Ex: 2 for 2D and 3 for 3D (int)  
  --epochs 	      	Number of training epochs (int)
						   
optional arguments:
  -h, --help            Show this help message and exit  
  --lr			Learning rate (int)
  --batch_size 		Batch_size (int)
  --use_mask 		Consider visibility mask (bool)
  --interactive 	Consider interaction between persons (bool)
  --persons_num 	Number of persons in each sequence (int)
  --snapshot_interval 	Save snapshot every N epochs (int)
  --load_path  		Path to load a model (str)
  --distance_loss 	Name of distance loss. (str)
  --mask_loss_weight	Weight of mask-loss (float)
  --decay_factor  	Decay_factor for learning_rate (float)
  --decay_patience 	Decay_patience for learning_rate (int)
  --skip_num 		Number of frames to skip in reading dataset (int)
  --start_epoch  	Start epoch (int)
  --shuffle 		Shuffle dataset (bool)
  --pin_memory		Pin memory (bool)
  --num_workers  	Number of workers (int)
  --n_layers		Number of layers for LSTMs (int)
  --hidden_size		Hidden size for LSTMs (int)
  --hardtanh_limit	Param for hardtanh activation function (float)
  --dropout_enc	     	Dropout rate for encoder (float)
  --dropout_pose_dec	Dropout rate for pose decoder (float)
  --dropout_mask_dec	Droput rate for mask decoder (float)
```  

Example:
```bash  
python -m api.train --train_dataset=<dataset_name> --model=<model_name> --keypoint_dim=2 --epochs=100 --interactive --persons_num=<perons_num>
```  
```bash  
python -m api.train --train_dataset=<dataset_name> --valid_dataset=<dataset_name> --model=<model_name> --keypoint_dim=2 --use_mask --epochs=100 --load_path=<path_to_model>
```  

## Evaluation

``` 
usage: python -m api.evaluate [-h] [--dataset] [--model] [--keypoint_dim] [--load_path]
                              	   [--use_mask] [--interactive] [--persons_num]
                              	   [--batch_size] [--distance_loss] [--skip_num]
                                   [--shuffle] [--pin_memory] [--num_workers] 

mandatory arguments:
  --dataset    		Name of dataset Ex: 'posetrack' or '3dpw' (str)  
  --model          	Name of model (str)  
  --keypoint_dim        Number of dim data should have Ex: 2 for 2D and 3 for 3D (int)  
  --load_path  		Path to load a model (str)
						   
optional arguments:
  -h, --help            Show this help message and exit  
  --use_mask 		Consider visibility mask (bool)
  --interactive 	Consider interaction between persons (bool)
  --persons_num 	Number of persons in each sequence (int)
  --batch_size 		Batch_size (int)
  --distance_loss 	Name of distance loss. (str)
  --skip_num 		Number of frames to skip in reading dataset (int)
  --shuffle 		Shuffle dataset (bool)
  --pin_memory		Pin memory (bool)
  --num_workers  	Number of workers (int)
```  

Example:
```bash  
python -m api.evaluate --dataset=<dataset_name> --model=<model_name> --keypoint_dim=2 --load_path=<path_to_model>
```  

## Prediction

```  
usage: python -m api.predict [-h] [--dataset] [--model] [--keypoint_dim] [--load_path]
                             	  [--use_mask] [--interactive] [--persons_num]
                             	  [--pred_frames_num] [--batch_size] [--skip_num]
                             	  [--shuffle] [--pin_memory] [--num_workers]
							 
mandatory arguments:
  --dataset        	Name of dataset eg: 'posetrack' or '3dpw' (str)  
  --model         	Name of desired model. Mandatory if load_path is None. (str)  
  --keypoint_dim        Number of dim data should have eg: 2 for 2D and 3 for 3D (int)  
  --load_path  		Path to load a model (str)
  --pred_frames_num 	Number of frames to predict. Mandatory if load_path is None. (int)
						   
optional arguments:
  -h, --help            Show this help message and exit  
  --use_mask 		Consider visibility mask (bool)
  --interactive 	Consider interaction between persons (bool)
  --persons_num 	Number of persons in each sequence (int)
  --batch_size 		Batch_size (int)
  --skip_num    	Number of frames to skip in reading dataset (int)
  --shuffle 		Shuffle dataset (bool)
  --pin_memory		Pin memory (bool)
  --num_workers  	Number of workers (int)
```  

Example:
```bash  
python -m api.predict --dataset=<dataset_name> --keypoint_dim=2 --load_path=<path_to_model> 
```  
```bash  
python -m api.predict --dataset=<dataset_name> --model=<model_name> --keypoint_dim=2 --pred_frames_num=<pred_frames_num>
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
    dataset_type 	    Name of using dataset. (str) (['somof_posetrack', 'posetrack', 'somof_3dpw', '3dpw', 'jta', 'jaad', 'pie', 'human3.6m', 'human3.6_walking'])  
    model        	    Name of desired model. (str) (['zero_vel','nearest_neighbor', 'pv_lstm', 'disentangled', 'derpof', 'history_repeats_itself', 'mix_and_match', 'comp_pred_vel', 'comp_pred_pose','comp_pred_center', 'comp_pred_root','trans_cvae','pv_lstm_noisy', 'comp_pred_vel_concat', 'v_lstm_noisy', 'p_lstm_noisy', 'pv_lstm_pro', 'keyplast'])  
    images_dir 		    Path to existing images on your local computer (str)
    showing             Indicates which images we want to show (dash(-) separated list) ([observed, future, predicted, completed])   
    index             Index of a sequence in dataset to visualize. (int)
    data.normalize      If data.normalize = True, we have to pass <data.metadata_path>. (bool)
    data.metadata_path  Path to metadata of dataset. (str)
    data.is_visualizing Essentially indicates if we are using visualizer, MUST be True for this part (bool) (default: True)
    
    
optional arguments:  
  load_path             Path to pretrained model. Mandatory if using a training-based model (str)  
  pred_frames_num 	Number of frames to predict. Mandatory if load_path is None. (int)
  interactive         To consider interaction or not. (bool)  
  use_mask            Use visibility mask if possible for dataloader. (bool)  
  skip_num     	Number of frame to skip between each two used frames. (int)
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
