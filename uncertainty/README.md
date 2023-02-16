Uncertainty in pose prediction
==============================

<div dir="ltr" align="justify">

## Abstract


### Install dependencies

```
pip install -r requirements.txt
```

### Dataset

* Human 3.6
  The main dataset which our experiments relies on; consisting of seven subjects.
  Directory structure:
  ```shell
  H3.6m
    |-- S1
    |-- S5
    |-- S6
    |-- ...
    |-- S11 
  ```
* AMAAS

### Training Arguments

* `n_clusters`: Number of the clusters. (default: 17)
* `dataset`: Name of the dataset so that the pipeline knows what sequence of joints should be omitted. (default: Human36m)
* 'input_n': Number of the input sequence. (default: 10)
* 'output_n': Number of the output sequence. (default: 25)

### Train

  ```batch
  python runner.py --dataset $DATASET_NAME --dataset_path $DATASET_PATH --output_path $OUT_PATH
  ```
  
### Test

  ```batch
  python runner.py --test true --dataset $DATASET_NAME --dataset_path $DATASET_PATH --dc_model_path $DC_MODEL_PATH --model_path $PREDICTION_MODEL_PATH
  ```

### Citing

### Acknowledgement

### License

</div>