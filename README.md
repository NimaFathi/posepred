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
|   ├── amass_preprocessor.py             -- preprocessing AMASS dataset class
|   ├── stanford_preprocessor.py             -- preprocessing Human3.6m dataset with stanford preprocessing class
    └── ...                             -- other datasets preprocessor class
├── data_loader
|   ├── solitary_dataset.py             -- handles dataloader for non-interactive data
|   ├── noisy_solitary_dataset.py       -- handles dataloader for noisy non-interactive data
|   └── interactive_dataset.py          -- handles dataloader for interactive data
|   └── random_crop_dataset.py          -- handles dataloader for general sequential data
├── factory
|   ├── trainer.py                      -- base code for training
│   ├── evaluator.py                    -- base code for evaluation 
|   └── output_generator.py             -- base code for testing
├── models                    
│   ├── history_repeats_itself/history_repeats_itself.py
|   ├── st_transformer/ST_Transformer.py
|   ├── pgbig/pgbig.py
│   ├── msr_gcn/msrgcn.py
|   ├── potr/potr.py
│   ├── sts_gcn/sts_gcn.py
|   ├── zero_vel.py
|   ├── pv_lstm.py
│   ├── disentangled.py
|   ├── derpof.py
|   ├── ...
├── losses
|   ├── pua_loss.py
│   ├── mpjpe.py           
|   ├── ...
├── metrics
|   ├── pose_metrics.py                     
│   └── mask_metrics.py            
├── optimizers
|   ├── sam.py
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

## MLFlow
We use MLflow in this library for tracking the training process. The features provided by MLFlow help users track their training process, set up experiments with multiple runs and compare runs with each other. Its clean, organized and beatiful UI helps users to better understand and track what they are doing: (you can see more from MLFlow in [here](https://mlflow.org/))

You can use MLFlow by running the command below in the folder containing `mlruns` folder.
```bash
mlflow ui
```
![img](https://www.mlflow.org/docs/latest/_images/tutorial-compare.png)

## Preprocessing
We need to create clean static file to enhance dataloader and speed-up other parts.  
To fulfill mentioned purpose You should run preprocessing api called `preprocess.py` like below:  

Example:  
```bash  
python -m api.preprocess \
    dataset=stanford3.6m \
    official_annotation_path=$DATASET_PATH \
    data_type=test
```  
See [here](https://github.com/vita-epfl/posepred/blob/master/ARGS_README.md#preprocessing) for more details about preprocessing arguments.
  
## Training
Train models from scratch:
```bash  
python -m api.train model=st_transformer \
          train_dataset=$DATASET_TRAIN_PATH \
          valid_dataset=$DATASET_VALIDATION_PATH \
          epochs=10 \
          snapshot_interval=2 \
          obs_frames_num=10 \
          pred_frames_num=25 \
          experiment_name=stTrans \
          data.seq_rate=5 \
          model.pre_post_process=human3.6m \
          model.n_major_joints=22
```  

Provide **validation_dataset** to adjust learning-rate and report metrics on validation-dataset as well.

See [here](https://github.com/vita-epfl/posepred/blob/master/ARGS_README.md#training) for more details about training arguments.


## Evaluation
Evaluate untrainable model:
```bash  
python -m api.evaluate model=zero_vel \
          dataset=$DATASET_TEST_PATH \
          obs_frames_num=10 \
          pred_frames_num=25 \
          data.is_testing=true \
          data.is_h36_testing=true
```  

evaluate pretrained model:
```bash
python -m api.evaluate model=st_transformer \
          dataset=$DATASET_TEST_PATH \
          load_path=$MODEL_CHECKPOINT \
          obs_frames_num=10 \
          pred_frames_num=25 \
          data.is_testing=true \
          data.is_h36_testing=true
```
See [here](https://github.com/vita-epfl/posepred/blob/master/ARGS_README.md#evaluation) for more details about evaluation arguments.


## Generating Outputs
Generate and save the predicted future poses:
```bash
python -m api.generate_final_output model=st_transformer \
          dataset=$DATASET_PATH \
          load_path=$MODEL_CHECKPOINT \
          obs_frames_num=5 \
          pred_frames_num=10 \
          data.is_h36_testing=true \
          save_dir=$OUTPUT_PATH
```  
See [here](https://github.com/vita-epfl/posepred/blob/master/ARGS_README.md#generating-outputs) for more details about prediction arguments.
  
  
## Visualization  
You can Visualize both 3D and 2D data with visualization module.  
See here for more details about visualization arguments. <br>
In order to generate .gif outputs you can run `visualize.py‍‍‍‍` like below:  
  
### 2D Visualization  
  
Example:  
```bash  
python -m api.visualize model=<model_name> dataset=<path_to_dataset> dataset_type=jta keypoint_dim=2 is_noisy=true data.noise_rate=0.1 device=cpu load_path=<path_to_model> index=25 showing=observed-future
```  
<!--   
Sample output:  
  
<div align="center">  
    <img src="visualization/outputs/2D/2D_visualize.gif" width="600px" alt><br>  
</div>   -->
  
### 3D Visualization  
  
If we have camera extrinsic and intrinsic parameters and image paths, we would create 2 gifs:  
- 2D overlay on images  
- 3D positions from the camera's POV  
  
Example:  
```bash  
python -m api.visualize model=st_transformer \
            dataset=$DATASET_PATH \
            dataset_type=stanford3.6m \
            data.len_observed=5 \
            data.len_future=10 \
            load_path=$MODEL_CHECKPOINT \
            index=50 \
            showing=observed-future-predicted \
            save_dir=$OUTPUT_PATH
```  
  
<!-- Sample outputs:  
<div align="center">  
   <!--  <img src="visualization/outputs/2D/3D_visualize_2D_overlay.gif" width="600px" alt><br>  -->
    <img src="visualization/outputs/3D/3D_visualize.gif" width="600px" alt>  
</div> -->

see [here](https://github.com/vita-epfl/posepred/blob/master/ARGS_README.md#visualization) for more details about visualization arguments.
