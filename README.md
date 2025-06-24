# AI-Care Binary Classification of Lung Cancer Survival
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14524261.svg)](https://doi.org/10.5281/zenodo.14524261)
[![DOI](https://img.shields.io/badge/Paper@MIE-10.3233/SHTI250379-blue)](doi.org/10.3233/SHTI250379)

This repository contains the implementation of a binary classification model for predicting survival outcomes in lung cancer patients from German cancer registry data. The project is part of the [AI-Care initiative](https://ai-care-cancer.de/). This work [has been published](https://doi.org/10.3233/SHTI250379) at the MIE 2025 conference.
To cite our paper, please use:
```
@incollection{germer2025lung,
  title={Lung Cancer Survival Estimation Using Data from Seven German Cancer Registries},
  author={Germer, Sebastian and Rudolph, Christiane and Katalinic, Alexander and Rath, Natalie and Rausch, Katharina and Handels, Heinz},
  booktitle={Intelligent Health Systems--From Technology to Data and Knowledge},
  pages={457--461},
  year={2025},
  doi={https://doi.org/10.3233/SHTI250379},
  publisher={IOS Press}
}
```

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
Then, run `podman run --rm --mount type=bind,source=./data_path,target=/app/data --mount type=bind,source=./result_path,target=/app/results aicare-binary-classification:latest` (adjust local paths to your needs!)


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
    --dummy, action="store_true": Use dummy classifier that always predicts the most frequent class
    --data_path, type=str: path to your data
    --entity, type=str: Entity to train on (lung, breast, thyroid, non_hodgkin_lymphoma)
    --traintestswap, action="store_true": Whether to use register as training data instead of test data
```


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