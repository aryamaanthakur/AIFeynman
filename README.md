# Evaluation Test: SYMBA

Project: [Evolutionary and Transformer Models for Symbolic Regression](https://ml4sci.org/gsoc/2024/proposal_SYMBA3.html)

This repository contains the solution to evaluation task of above project. Below is my approach towards the problem.

Model weights can be found [here](https://drive.google.com/file/d/1rUqJLPFTxfv5cCBHWXTh8poMqxwMUK5Q/view?usp=sharing)

## Tasks

The evaluation test consisted of 3 tasks (2 common and 1 specific). Following is my approach to solving the tasks and rationale behind choice of methods.

### Task 1
`dataset.py` contains the framework for converting original dataset into a easy-to-use form. I have written custom tokenizers for both floats and equations. Files related tokenization of data could be found in `utils` directory.

Pre-processing consists of 2 parts

#### Input

- Input consists of floating point numbers where last number is y. Firstly, last column is appended to the beginning to ensure dependent variable always stays in the first column

- The floating numbers are tokenized using P1000 (François Charton). In this method, the mantissa is encoded as a single token (from 0 to 999) and a number is represented as the triplet (sign, mantissa, exponent).

- Each equation contains 1000000 datapoints stored in a single file. This file is opened, tokenized and divided into chunks of 400 datapoints and saved as .npy file in the corresponding folder.

#### Output
- Output consists of equations. Firstly the equations are cleaned and converted to sympy expressions.

- The variables in the equations are replaces by generic variable identifiers (e.g. s_1, s_2, s_3, etc.)

- Then these equations are converted to prefix form (direct polish notation)

- After tokenizing the prefix equations, these are saved in a .npy file


Dataset is divided into train set and test set such that both are mutually exclusive.
Then the train set is further divided into train and validation set but here both subsets have same equations and different datapoints.

### Task 2

I trained a generic seq2seq transformer model (Vaswani et al.)with 2 encoders and 6 decoders. Number of decoders is more because generation is more intricate in this situation.

Input tokens are not being used directly. Encoding 400 datapoints where each datapoint contains 10 numbers would require `400*10*3 = 12000` tokens. Since, this is not possible in normal transformers, the input (`N*n*3`) is passed through 2 fully-connected layers and an activation function to output 1 embedding for each datapoint (`N*dim`).

Model was trained using cross-entropy (next token prediction loss). The sequence accuracy appears very low but I am quite sure its because the model is being trained only on 90 equations are predicting 10 completely new equations is a hard task in this situation. Using more data will solve this problem without a doubt.

### Task 3

I found a PyTorch implementation of evolved transformer (So et al.) [here](https://github.com/moon23k/Transformer_Variants/blob/main/model/evolved.py) and included it in this library. I trained a hybrid version of this model using next token prediction. Again, sequence accuracy appears low because amount of data was very less.

## Repository Structure

```
.
|-- engine/
|   |-- __init__.py
|   |-- config.py
|   |-- predictor.py
|   |-- trainer.py
|   `-- utils.py
|-- model/
|   |-- __init__.py
|   |-- evolved_transformer.py
|   `-- seq2seq.py
|-- utils/
|   |-- float_sequence.py
|   |-- sympy_prefix.py
|   `-- tokenizers.py
|-- dataset.py
|-- decoder_vocab
|-- encoder_vocab
|-- FeynmanEquationsModified.csv
|-- README.md
|-- task1-final.ipynb
|-- task2-final.ipynb
`-- task3-final.ipynb
```

## Instructions

### Installation
Clone the repository

`git clone https://github.com/aryamaanthakur/AIFeynman.git`

### Dataset Preparation

Step 1: Create a configuration
```python
class Config:
    input_max_len = 1000
    max_len = 11
    df_path = "./FeynmanEquationsModified.csv"
    output_dir = "./data"
    encoder_vocab = "./encoder_vocab"
    decoder_vocab = "./decoder_vocab"
```

Step 2: Call `prepare_dataset` function
```python
from dataset import prepare_dataset
train_df, equations_df = prepare_dataset(Config)
```

Step 3: Save the returned dataframes

### Training

Step 1: Load datasets and dataloaders. Pass the [train, valid, test] split as an argument in `get_datasets` function. Pass batchsizes of train, valid and test set as arguments in `get_dataloaders` function.
```python
import pandas as pd
from dataset import get_dataloaders, get_datasets

df = pd.read_csv("FeynmanEquationsModified.csv")
input_df = pd.read_csv("./data_400/train_df.csv")

datasets, train_equations, test_equations = get_datasets(df, input_df, "./data_400", [0.80, 0.1, 0.1])

dataloaders = get_dataloaders(datasets, 512, 512, 100)
```

Step 2: Create a configuration

```python
from engine import Config
config = Config(experiment_name="evolved",
                model_name="evolved_transformer",
                epochs = 10,
                optimizer_lr = 0.0005,
                optimizer_weight_decay = 0.0001,
                T_0 = 10,
                num_encoder_layers = 2,
                num_decoder_layers = 6,
                input_emb_size = 64,
                embedding_size = 64,
                hidden_dim = 64,
                pff_dim = 128,
                device = "cuda:6")
```

Step 3: Initialize and run the trainer

```python
from engine import Trainertrainer = Trainer(config, dataloaders)
trainer.train()

trainer.test_seq_acc() # to get sequence accuracy on test set
```

## Contact
For any questions or issues regarding this repository, please contact `aryamaanthakur.2002@gmail.com`