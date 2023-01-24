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

### Modes

The code can operate in three modes which is given by the argument `pipeline`
* `main`: This pipeline is executed if the uncertainty for a prediction model is needed. Note that all `Human 3.6` actions are included.
* `divided`: Equivalent pipeline for divided experiment. Excluding `smoking`, `phoning` and`takingphoto` from other actions then executing the main one.
* `rejection`: Use this pipeline to calculate the rejection rate and self-uncertainty rate for a given model.

### Arguments

* `fake_labeling`: Determines whether fake samples are required.
* `n_clusters`: Number of the clusters.
* `fake_clusters`: If the code has been executed in `main` pipeline with fake sample generation, mentioning the clusters with mainly fake data is needed.
* `model_path`: The path to a pose prediction model.
* `model_dict_path`: The path to dictionary of a pose prediction model; containing ground truth and output tensors. Note that a dictionary is not suitable for the `rejection` pipeline.

### Train

* Main
  ```batch
  python --pipeline main
  ```
* Main (with fake samples)
  ```batch
  python --pipeline main --fake_labeling True --n_clusters 20
  ```
* Divided
  ```batch
  python --pipeline divided
  ```

### Test

* Main
  ```batch
  python --pipeline main --dc_path /path/to/dc_model --ae_path /path/to/final_ae_model --model_dict_path ./prediction/dict/sts.pt
  ```
* Main (with fake samples)
  ```batch
  python --pipeline main --fake_labeling True --n_clusters 20 --model_dict_path ./prediction/dict/sts.pt
  ```
* Divided
  ```batch
  python --pipeline divided --dc_path /path/to/dc_model/ --ae-path /path/to/final_ae_model --model_dict ./prediction/model/sts_12_act.pt
  ```
* Rejection
  ```batch
  python --pipeline rejection --dc_path ./pretrained/dc_main.pt --ae_path ./pretrained/final_ae_main.pt
  ```

### Citing

### Acknowledgement

### License

</div>