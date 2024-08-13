### TS-CVAE: Accurate Player Movement Predictions by Leveraging Team and Opponent Dynamics for Doubles Badminton

## Overview
TS-CVAE incorporates Team GAT, which leverages team influence graph over a few strokes to capture rapidly changing team strategies, and Opponent GAT, which holistically analyzes interactions between opposing players.
## Setup

1. Clone the repo
    ```bash
    git clone https://github.com/spj-as/TS-CVAE.git
    ```
2. Create virtual env
   ```bash
   python3 -m venv venv
   ```
3. Activate virtual env
   ```bash
   source venv/bin/activate
   ```
4. Install requirements
   ```bash
   pip install -r requirements.txt
   ```
5. Generate train and test data
   ```bash
   python main.py preprocess
   ```

## Train

1. Train gagnet model with graph attention and default parameters
    ```bash
   
    ```

## Dataset Description

> You should run `python main.py preprocess` to generate train and test data. The data will be saved in `data` folder.

- Shape of train data
  
  displacements
      (train_len, 12, 4, 2)
  
  velocity
      (train_len, 12, 4, 2)
  
  goals (shot type)
      (train_len, 12, 4, 16)
  
  hit (hitting player)
      (train_len, 12, 4, 4)
  
- Shape of test data
  
  displacements
      (test_len, 12, 4, 2)
  
  velocity
      (test_len, 12, 4, 2)
  
  goals (shot type)
      (test_len, 12, 4, 16)
  
  hit (hitting player)
      (test_len, 12, 4, 4)
  
