# Posepred  
Posepred is an open-source toolbox for pose prediction/forecasting a sequence of human pose given an observed sequence, implemented in PyTorch.

<p float="left">
  Input pose<br/><img src="https://user-images.githubusercontent.com/33596552/138102745-f6b5c7a0-ee14-40ef-907f-b3ebb98ae08f.gif" alt="observation" width="500">

  Output pose and ground-truth<br/><img src="https://user-images.githubusercontent.com/33596552/138102754-5bef72df-ea48-4d17-a932-611293f0bc5a.gif" alt="prediction" width="500">  
</p>

# Overview 
The main parts of the library are as follows:

```
posepred
├── api
|   ├── preprocess.py                   -- script to run the preprocessor module
|   ├── train.py                        -- script to train the models, runs factory.trainer.py
│   ├── evaluate.py                     -- script to evaluate the models, runs factory.evaluator.py
|   └── generate_final_output.py        -- script to generate and save the outputs of the models, runs factory.output_generator.py
|   ├── visualize.py                    -- script to run the visualization module
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
├── uncertainty
|   ├── main.py
|   ├── runner.py
├── losses
|   ├── pua_loss.py
│   ├── mpjpe.py           
|   ├── ...
```
The library has 5 important API which 1) preprocess data 2) train the model 3) evaluate it quantitatively 4) generate their outputs and 5) visualize it. The details of how to use these API are described below. Two other important directories are models and losses. In these two directories, you can add any desired model and loss function and leverage all predefined functions of the library to train and test and compare in a fair manner.

Please check other directories (optimizers, mmetrics, schedulers, visualization, utils, etc.) for more abilites.

# Getting Started  
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
Before moving forward, you need to install Hydra and know its basic functions to run different modules and APIs.  
hydra is A framework for elegantly configuring complex applications with hierarchical structure.
For more information about Hydra, read their official page [documentation](https://hydra.cc/).

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

# Preprocessing
We need to create clean static file to enhance dataloader and speed-up other parts.
To fulfill mentioned purpose, put the data in DATASET_PATH and run preprocessing api called `preprocess.py` like below:  

Example:  
```bash  
python -m api.preprocess \
    dataset=stanford3.6m \
    official_annotation_path=$DATASET_PATH \
    data_type=test
```  
See [here](https://github.com/vita-epfl/posepred/blob/master/ARGS_README.md#preprocessing) for more details about preprocessing arguments.
This process should be repeated for training, validation and test set. This is a one-time use api and later you just use the saved jsonl files.
  
# Training
Given the preprocessed data, train models from scratch:
```bash  
python -m api.train model=st_transformer \
    train_dataset=$DATASET_TRAIN_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    epochs=4 \
    snapshot_interval=2 \
    obs_frames_num=5 \
    pred_frames_num=10 \
    model.loss.nT=10 \
    experiment_name=stTrans \
    data.seq_rate=5 \
    model.pre_post_process=human3.6m \
    model.n_major_joints=22 \
    model.loss.nJ=32
```  

**NOTE**: You can see more commands for training models [here](COMMANDS.md).

Provide **validation_dataset** to adjust learning-rate and report metrics on validation-dataset as well.

See [here](https://github.com/vita-epfl/posepred/blob/master/ARGS_README.md#training) for more details about training arguments.


# Evaluation
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
evaluate model with uncertainty being calculated:
```bash
python -m api.evaluate model=st_transformer \
          dataset=$DATASET_TEST_PATH \
          load_path=$MODEL_CHECKPOINT \
          obs_frames_num=10 \
          pred_frames_num=25 \
          data.is_testing=true \
          data.is_h36_testing=true \
          eval_uncertainty=true \
          oodu_load_path=$UNCERTAINTY_MODEL
```
See [here](https://github.com/vita-epfl/posepred/blob/master/ARGS_README.md#evaluation) for more details about evaluation arguments.

See [here](https://github.com/vita-epfl/posepred/blob/uncertainty-g1/uncertainty/README.md) if training the uncertainty calculator model is needed.


# Generating Outputs
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
  
  
# Visualization  
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
- 3D positions from the camera's point of view  
  
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
