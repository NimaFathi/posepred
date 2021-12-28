# Posepred  
Pospred is an open-source toolbox for pose prediction based on PyTorch. It is a part of the VitaLab project.    

<p float="left">
  <img src="https://user-images.githubusercontent.com/33596552/138102745-f6b5c7a0-ee14-40ef-907f-b3ebb98ae08f.gif" alt="prediction" width="500">  
  <img src="https://user-images.githubusercontent.com/33596552/138102754-5bef72df-ea48-4d17-a932-611293f0bc5a.gif" alt="observation" width="500">  
</p>


```
posepred
├── api
|   ├── preprocess.py                   -- script to run the preprocessor module
|   ├── train.py                        -- script to train the models, runs factory.trainer.py
│   ├── evaluate.py                     -- script to evaluate the models, runs factory.evaluator.py
|   ├── visualize.py                    -- script to run the visualization module
|   └── generate_final_output.py        -- script to generate and save the outputs of the models, runs factory.output_generator.py
├── preprocessor
|   ├── preprocessor.py                 -- base class for preprocessor module
|   ├── dpw_preprocessor.py             -- preprocessing 3DPW dataset class
    └── ...                             -- other datasets preprocessor class
├── data_loader
|   ├── solitary_dataset.py             -- handles dataloader for non-interactive data
|   ├── noisy_solitary_dataset.py       -- handles dataloader for noisy non-interactive data
|   └── interactive_dataset.py          -- handles dataloader for interactive data
├── factory
|   ├── trainer.py                      -- base code for training
│   ├── evaluator.py                    -- base code for evaluation 
|   └── output_generator.py             -- base code for testing
├── models
|   ├── pv_lstml.py                     
│   ├── neareset_neighbor.py            
|   ├── zero_vel.py                     
|   ├── ...
├── losses
|   ├── kl_divergence.py                     
│   ├── mae_vel.py            
|   ├── ...
├── metrics
|   ├── pose_metrics.py                     
│   └── mask_metrics.py            
├── optimizers
|   ├── adam.py                     
|   ├── sgd.py            
|   ├── ...
├── schedulers
|   ├── reduce_lr_on_plateau.py
|   ├── step_lr.py            
|   ├── ...
├── utils
|   ├── average_meter                   -- updating average for metrics each epoch
|   ├── reporter.py                     -- to calcualte and report losses and metrics
|   ├── save_load.py                    -- base code for saving and loading models
|   └── others.py                       -- other useful utils
└── visualization
    ├── color_generator.py              -- code for generating miscellanoeus colors
    ├── keypoints_connection.py         -- dictionary for creating joints connection graphs
    └── visualizer.py                   -- base code for visualization

```

## Installation  
To get started as quickly as possible, follow the instructions in this section. This should allow you train a model from scratch, evaluate your pretrained models, and produce basic visualizations.  
  
### Dependencies  
Make sure you have the following dependencies installed before proceeding:  
- Python 3.7+ distribution  
- PyTorch >= 1.7.0  
- CUDA >= 10.0.0 
- pip >= 21.3.1 
### Virtualenv  
You can create and activate virtual environment like below:  
```bash  

pip install --upgrade virtualenv

virtualenv -p python3.7 <venvname>  

source <venvname>/bin/activate  

pip install --upgrade pip
```  
### Requirements  
Furthermore, you just have to install all the packages you need:  
  
```bash  
pip install -r requirements.txt  
```  
Before moving forward you need to install Hydra and know its basic functions to run different modules and APIs.  
hydra is A framework for elegantly configuring complex applications with hierarchical structure.

To understand Hydra better read the official [documentation](https://hydra.cc/). It is not essential stage  to work with our library, but we encourage you to do.

## Hydra
In order to have a better structure and understanding of our arguments, we use Hydra to  dynamically create a hierarchical configuration by composition and override it through config files and the command line.
If you have any issues and errors install hydra like below:
```bash
pip install hydra-core --upgrade
```
for more information about Hydra and modules please visit [here](https://github.com/vita-epfl/posepred/blob/master/ARGS_README.md#Hydra)

## Preprocessing
We need to create clean static file to enhance dataloader and speed-up other parts.  
To fulfill mentioned purpose You should run preprocessing api called `preprocess.py` like below:  

Example:  
```bash  
python -m api.preprocess dataset=somof_3dpw official_annotation_path=<path_to_dataset> data_type=train keypoint_dim=3 skip_num=1 obs_frames_num=16 pred_frames_num=14 interactive=false
```  
See [here](https://github.com/vita-epfl/posepred/blob/master/ARGS_README.md#preprocessing) for more details about preprocessing arguments.
  
## Training
Train models from scratch:
```bash  
python -m api.train model=<model_name> keypoint_dim=3 train_dataset=<path_to_dataset> valid_dataset=<path_to_dataset> epochs=250 data.shuffle=True device=gpu snapshot_interval=10 hydra.run.dir=<path_to_output>
```  

Provide **validation_dataset** to adjust learning-rate and report metrics on validation-dataset as well.

See [here](https://github.com/vita-epfl/posepred/blob/master/ARGS_README.md#training) for more details about training arguments.


## Evaluation
Evaluate pretrained model:
```bash  
python -m api.evaluate model=<model_name> dataset=<path_to_dataset> keypoint_dim=3 is_noisy=True data.shuffle=True rounds_num=5 data.noise_rate=0.2 load_path=<path_to_model> device=cpu
```  
See [here](https://github.com/vita-epfl/posepred/blob/master/ARGS_README.md#evaluation) for more details about evaluation arguments.


## Generating Outputs
Generate and save the predicted future poses:
```bash  
python -m api.generate_final_output dataset=<dataset_name> model=<model_name> keypoint_dim=2 load_path=<path_to_model> data.is_interactive=False device=cpu pred_frames_num=<int> 
```  
See [here](https://github.com/vita-epfl/posepred/blob/master/ARGS_README.md#generating-outputs) for more details about prediction arguments.
  
  
## Visualization  
You can Visualize both 3D and 2D data with visualization module.  
See here for more details about visualization arguments. <br>
In order to generate .gif outputs you can run `visualize.py‍‍‍‍` like below:  
  
### 2D Visualization  
  
Example:  
```bash  
python -m api.visualize model=<model_name> dataset=<path_to_dataset> dataset_type=jta keypoint_dim=2 is_noisy=true load_path=<path_to_model> index=25 showing=observed-future
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
python -m api.visualize model=<model_name> dataset=<path_to_dataset> dataset_type=jta keypoint_dim=3 is_noisy=true load_path=<path_to_model> index=25 showing=observed-future
```  
  
Sample outputs:  
<div align="center">  
   <!--  <img src="visualization/outputs/2D/3D_visualize_2D_overlay.gif" width="600px" alt><br>  -->
    <img src="visualization/outputs/3D/3D_visualize.gif" width="600px" alt>  
</div>

see [here](https://github.com/vita-epfl/posepred/blob/master/ARGS_README.md#visualization) for more details about visualization arguments.
