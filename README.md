# Trans-SVNet: Accurate Phase Recognition from Surgical Videos via Hybrid Embedding Aggregation Transformer

This is a fork of the offical code repo at https://github.com/xjgaocs/Trans-SVNet. This repo provides additional code and instructions to reproduce the results in the [paper](https://arxiv.org/abs/2103.09712). Large parts of the additional code are based on the [TMRNet](https://arxiv.org/abs/2103.16327) implementation, see https://github.com/YuemingJin/TMRNet.

## Getting started

- Clone this repository recursively:
```
git clone --recursive https://github.com/IsabelFunke/Trans-SVNet.git
```
- Download the files that the authors provide in the [Google Drive](https://drive.google.com/drive/folders/1Zgrg1G8fdrOzQ5W3FREl35NmDQMDWEXp?usp=sharing). Copy the files to the following locations in the `Trans-SVNet` directory:
    ```
    <code_dir>
        Trans-SVNet
            best_model
                emd_lr5e-4
                    resnetfc_ce_epoch_15_length_1_opt_0_mulopt_1_flip_1_crop_1_batch_100_train_9946_val_8404_test_7961.pth
                TeCNO
                    TeCNO50_epoch_6_train_9935_val_8924_test_8603.pth
                    TeCNO50_trans1_3_5_1_length_30_epoch_0_train_8769_val_9054.pth
            ...
            train_val_paths_labels1.pkl
            trans_SV.py
            ...
    ```
- Download the [Cholec80 dataset](http://camma.u-strasbg.fr/datasets), unzip it and store it at some location (`<data_root>`). The expected folder structure looks like this:
    ```
    <data_root>
        cholec80
            phase_annotations
                video01-phase.txt
                video02-phase.txt
                ...
            videos
                video01.mp4
                video02.mp4
                ...
            ...
    ```
- Activate Python environment
We provide all used Python packages in `environment.yml`. Anaconda can be used to recreate our Python environment:
    ```
    cd <code_dir>/Trans-SVNet
    */Trans-SVNet$ conda env create -f environment.yml
    */Trans-SVNet$ conda activate torch151
    ```


## Data preprocessing

- Adjust lines 15 and 16 in `video2frame_cutmargin.py` to:
    ```
    source_path = "<data_root>/cholec80/videos/"  # original path
    save_path = "<data_root>/cholec80/cutMargin/"  # save path
    ```
    (replace `<data_root>` with the correct path to your Cholec80 dataset)
- Run
    ```
    (torch151) */Trans-SVNet$ python video2frame_cutmargin.py
    ```
    (this will take a while...)
- The provided code expects to find the folder `cutMargin` relative to the `Trans-SVNet` folder at `../../Dataset/cholec80/cutMargin`. You can move and rename the folder at `<data_root>` accordingly. Alternatively, you can create a symbolic link:
    - Create the folder `Dataset` such that it is a sibling of the direct parent folder of `Trans-SVNet`
    - Change to the direct parent folder of `Dataset`
    - Run `ln -s <data_root>/cholec80 Dataset`

    Finally, the folder structure needs to look like
    ```
    <some dir>
        Dataset
            cholec80
                cutMargin
                    1
                        0.jpg
                        25.jpg
                        ...
                    2
                    ...
        <parent>
            Trans-SVNet
                ...
                trans_SV.py
                ...
    ```

## Reproduce results using the provided trained models

### 1. Extract ResNet50 features
```
(torch151) */Trans-SVNet$ python generate_LFB.py --skip_train
```
The extracted features will be stored in pickle files in the directory `LFB`. The option `--skip_train` means that features will only be extracted for the validation and test data in order to save some computation time.
### 2. Get Trans-SVNet predictions
```
(torch151) */Trans-SVNet$ python test_trans_SV.py
```
The following files will be generated:
- `Eval/Test_Trans-SVNet/40-8-32/-/predictions.yaml`
    This file contains the predictions on the 32 test videos in a human-readable format.
- `Eval/Test_Trans-SVNet/40-40/-/predictions.yaml`
    This file contains the predictions on the 8 validation videos and the 32 test videos in a human-readable format.
- `Eval/Test_Trans-SVNet/all_predictions.pkl`
    This pickle file contains the predictions on the 8 validation videos and the 32 test videos, so 40 videos in total.
### 3a. Compute evaluation metrics with relaxed boundaries
This is based on evaluation code from the TMRNet repository.
- Change to the subdirectory `Eval/relaxed_metrics`
- Generate required helper files
    - Adjust line 7 in `get_paths_labels.py` to `root_dir2 = "<data_root>/cholec80"`, where `<data_root>` equals the path to your Cholec80 dataset.
    - Execute the script:
        ```
        (torch151) */Trans-SVNet/Eval/relaxed_metrics$ python get_paths_labels.py
        ```
    - Convert the predictions in `Eval/Test_Trans-SVNet/all_predictions.pkl` into `video*-phase.txt` files by running:
             ```
             (torch151) */Trans-SVNet/Eval/relaxed_metrics$ python export_phase_copy.py --name "../Test_Trans-SVNet/all_predictions.pkl"
             ```
- Run the MATLAB evaluation script. We used [GNU Octave](https://octave.org/), see [this Readme](https://gitlab.com/nct_tso_public/phasemetrics#installing-octave-and-oct2py-optional) for installation extractions (an installation of `oct2py` is not required).
    ```
    */Trans-SVNet/Eval/relaxed_metrics$ cd matlab-eval
    */Trans-SVNet/Eval/relaxed_metrics/matlab-eval$ octave
    octave:1> pkg load image; pkg load statistics; pkg load io
    octave:2> Main
    ...
    octave:3> exit
    ```
    The evaluation results will be printed to the Octave shell.
### 3b. Compute regular evaluation metrics
This is based on the implementation at https://gitlab.com/nct_tso_public/phasemetrics.
- Change to subdirectory `Eval`
- Run the evaluation script, where `<data_root>` needs to be replaced with the correct path to the Cholec80 data. The option `--datasplit` can also be `"40-40"` in order to compute the results on all 40 videos (32 test + 8 validation).
        ```
        (torch151) */Trans-SVNet/Eval$ python -m PhaseMetrics.eval --experiment "Test_Trans-SVNet" --datasplit "40-8-32" --results_root "." --data_root "<data_root>/cholec80"
        ```
- An evaluation report with the computed evaluation metrics will be created at `Eval/Test_Trans-SVNet/40-8-32\eval.yaml`. See the [PhaseMetrics repo](https://gitlab.com/nct_tso_public/phasemetrics)  for further documentation.


## Credits
This is a fork of the offical code repo at https://github.com/xjgaocs/Trans-SVNet.
The following files where adjusted from https://github.com/YuemingJin/TMRNet:
- `video2frame_cutmargin.py`
- All scripts in `Eval/relaxed_metrics` 