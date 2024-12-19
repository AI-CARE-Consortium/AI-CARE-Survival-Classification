# AI-Care Binary Classification of Lung Cancer Survival
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13986103.svg)](https://doi.org/10.5281/zenodo.13986103)

This repository contains the implementation of a binary classification model using `main.py`.

## Requirements
- After cloning, you need to load the data_import submodule. This can be done using:
```bash
git submodule init
git submodule update
```

- Created with Python 3.12.4
- Used libraries are scikit-learn, catboost, optuna and pandas.
  You can create such a enviroment with the [enviroment.yml](enviroment.yml) or by running via Docker/Podman.

## Usage
### Via Docker / Podman
After loading the submodule, run `podman build -t localhost/aicare-binary-classification:latest .` to build the image.
Then, run `podman run --rm --mount type=bind,source=./data_path,target=/app/data --mount type=bind,source=./results,target=/app/results aicare_binary_classification:latest` (adjust source paths to your needs!)


### Manually
To run the binary classification model, execute the following command:

```bash
python main.py 
```
with the following arguments:
```
    --registry, type=str: registry number according to   
    --months, type=int: 'Survival months to binary classify'
    --inverse, action="store_true": Inverse the binary classification
    --dummy, action="store_true", Use dummy classifier that always predicts the most frequent class
    --data_path, type=str: path to your data
    --entity, type=str, Entity to train on (lung, breast, thyroid, non_hodgkin_lymphoma)
```




## main.py

The `main.py` script is the main entry point for training and evaluating the binary classification model. It includes the following key functionalities:

- Data loading and preprocessing
- Model definition and compilation
- Training the model
- Evaluating the model performance

## Directory Structure

```
│
├── main.py
├── README.md
├── Dockerfile
├── run.sh
├── environment.yml
└── data_import
    ├── data_preprocessing.py
    └── data_loading.py
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or issues, please contact [sebastian.germer@dfki.de](mailto:sebastian.germer@dfki.de).