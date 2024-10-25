### TS-CVAE: Accurate Player Movement Predictions by Leveraging Team and Opponent Dynamics in Doubles Badminton

## Overview
TS-CVAE incorporates two key components to enhance prediction accuracy: **Team GAT** and **Opponent GAT**. The **Team GAT** captures rapidly changing team strategies by leveraging a team influence graph. Meanwhile, the **Opponent GAT** holistically analyzes interactions between opposing players to understand the dynamics in doubles badminton.

## Setup

1. **Clone the repository**
    ```bash
    git clone https://github.com/spj-as/TS-CVAE.git
    ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment**
   ```bash
   source venv/bin/activate
   ```

4. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

5. **Generate training and testing data**
   ```bash
   python main.py preprocess
   ```

## Training the Model

1. **Train the model with default parameters**
    ```bash
    python main.py tscvae --name experiment_1 
    ```

## Dataset Description

After running `python main.py preprocess`, the generated training and testing datasets will be saved in the `data` folder. The datasets have the following shapes:

- **Training Data:**
  
  - **Displacements**: `(train_len, 12, 4, 2)`
  
  - **Velocity**: `(train_len, 12, 4, 2)`
  
  - **Goals (Shot Type)**: `(train_len, 12, 4, 16)`
  
  - **Hit (Hitting Player)**: `(train_len, 12, 4, 4)`

- **Testing Data:**

  - **Displacements**: `(test_len, 12, 4, 2)`
  
  - **Velocity**: `(test_len, 12, 4, 2)`
  
  - **Goals (Shot Type)**: `(test_len, 12, 4, 16)`
  
  - **Hit (Hitting Player)**: `(test_len, 12, 4, 4)`

## Randomness Handling

To introduce controlled randomness, the `Normal` class's `rsample` method is utilized for sampling:

```python
class Normal:
    def __init__(self, mu=None, logvar=None, params=None):
        super().__init__()
        if params is not None:
            self.mu, self.logvar = torch.chunk(params, chunks=2, dim=-1)
        else:
            assert mu is not None
            assert logvar is not None
            self.mu = mu
            self.logvar = logvar
        self.sigma = torch.exp(0.5 * self.logvar)

    def rsample(self):
        eps = torch.randn_like(self.sigma)
        return self.mu + eps * self.sigma

    def sample(self):
        return self.rsample()
```
## Experiment Setting 
In the framework, the dimensions of the hidden states and the latent variables are 8 and 16, respectively. The learning rate is set to 0.001, and the batch size is 32. Lambda is set to 1. For pretraining, the learning rate is set to 0.0005, and the batch size is 1024. All baselines are trained on a machine with an AMD Ryzen 7 5700X 8-Core CPU and an Nvidia RTX 3060 Ti GPU.
