# Sentiment Analysis Classification

## Getting Started

This project showcases an example application of PyTorch Lightning for sentiment analysis classification.


## Installation

The project is developed using Python 3.10. To install the dependencies needed for the project, run the following
commands:

1. `pip install pdm`
2. `pdm sync`
3. Activate the virtual environment

## Training the model

To train the model you can run the following command:

```
pdm run dl
```

Otherwise, if you want to use the training script directly, you can run the following command from the root directory of the project:

```
python src/dl/scripts/train.py
```


## Visualizing the model results

The results of the training and testing experiments are stored in the `.experiments` folder. You can view the results of these experiments by
using tensorboard. To do so, run the following command:

```
pdm run tensorboard
```
