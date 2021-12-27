# Arguments
This is a description to provide details about arguments of Posepred API.
Pospred is an open-source toolbox for pose prediction in PyTorch. It is a part of the VitaLab project.  
  
## Preprocessing   
Preprocessing config file: "configs/hydra/preprocess.yaml"
Also you can change preprocessor via commandline like below:
```  
usage: python -m api.preprocess  [official_annotation_path] [dataset] [data_type]                          	
	                             [obs_frames_num] [pred_frames_num] [use_video_once]
				                 [annotate_openpifpaf] [image_dir] [keypoint_dim]  
	                             [skip_num] [interactive] [output_name]
  
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
  --annotaion		Implies that to use dataset_path as annotation path or a path to images. in this context, we generate annotations using openpifpaf then 			create static files afterward. 
  
  --skip_num        	Number of frame to skip between each two used frames (int)
  --use_video_once      Use whole video just once or use until last frame (bool)  
  --output_name         Name of generated csv file (str)
  --annotation
```  
Example:  
```bash  
python -m api.preprocess --dataset_name=<dataset_name> --dataset_path=<path_to_dataset> --data_usage='train' --obs_frames_num=16 --pred_frames_num=14 --use_mask  
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

```  
usage: python -m api.visualize [-h] [--dataset] [--model] [--keypoint_dim] [--images_dir]
                               	    [--persons_num] [--index] [--load_path] [--ground_truth]
                              	    [--pred_frames_num] [--interactive] [--use_mask][--skip_num]  
  
mandatory arguments:  
  --dataset 	    	Name of using dataset. (str)  
  --model        	Name of desired model. (str)  
  --keypoint_dim      	Number of dim data should have Ex: 2 for 2D and 3 for 3D (int)  
  --images_dir 		Path to existing images on your local computer (str)

    
optional arguments:  
  -h, --help            Show this help message and exit.  
  --index               Index of a sequence in dataset to visualize. (int)  
  --ground_truth        Visualize ground-truth future frames as well. (bool)  
  --load_path           Path to pretrained model. (str)  
  --pred_frames_num 	Number of frames to predict. Mandatory if load_path is None. (int)
  --interactive         To consider interaction or not. (bool)  
  --persons_num         Number of people in each sequence. Mandatory if interactive is selected. (int)
  --use_mask            Use visibility mask if possible for dataloader. (bool)  
  --skip_num     	Number of frame to skip between each two used frames. (int)
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
