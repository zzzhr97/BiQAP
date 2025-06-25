# BiQAP

## Dependency

- NVIDIA GPU + CUDA
- Python 3.8

Please refer to the `requirements.txt` file to install the necessary packages.

## Dataset

Download the following datasets and place them in the `data` folder.
- [Graph Matching](https://drive.google.com/file/d/1N7INEIOEzBl0SH-_Vs3LPNjdAicSuC6o/view?usp=sharing)
- [Graph Edit Distance](https://drive.google.com/file/d/1Pdx3XFmD24337CUa8fb1j3ETzh_Mh1Xu/view?usp=sharing)
- [Traveling Salesman Problem](https://drive.google.com/file/d/1ALsbZDI0eITV1coxXrlN1n8xAwcktEAH/view?usp=sharing)
- [QAPLIB](https://drive.google.com/file/d/1v5p2PlySSJj20ihfYKVuVOEuQOLnQz5U/view?usp=sharing)

The folder structure is like:
```
- data/
    - GED/
        - AIDS/
        - .../
    - GM/
        - GM1/
        - .../
    - QAPLIB/
        - raw/
    - TSP/
        - tsp50/
```

For the large random dataset, please navigate to the `data` folder and run `python datagen.py`. This will generate the `L500`, `L750`, and `L1000` datasets. You can modify the `datagen.json` file to create datasets with different settings.

## Running

### Config

The config files are located in the `options` folder. We have included some files in this folder, and you can modify their parameters as needed.

### Training

Use the following command to train the models:

```bash
python main.py --mode train --cfg options/[your_config_training_file].yml
```

Note that during training, evaluation will be performed every pre-defined number of steps.

### Testing

Use the following command to test the models. This can be used for standalone evaluation of a specific model's weights or for testing a learning-free algorithm:

```bash
python main.py --mode test --cfg options/[your_config_testing_file].yml
```

### Results
The results will be saved in a specific directory within the `experiments` directory. You can find the exact folder based on the dataset, model name, and timestamp of the run. Inside the folder, you will find the config file used, model weights, and log files.