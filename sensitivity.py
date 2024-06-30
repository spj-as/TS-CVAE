import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ewma(return_list, alpha):
    ewma_values = [return_list[0]]  
    for episode_return in return_list[1:]:
        ewma_values.append(alpha * episode_return + (1 - alpha) * ewma_values[-1])
    return ewma_values

df = pd.read_csv('sensitivity.csv', skipinitialspace=True)

df.columns = df.columns.str.strip()

# Set smoothing factor
alpha = 0.09

iterations = np.array(range(1, 6))

models = df['Model'].unique()  
results = {}
for model in models:
    rewards = []
    for seed in df['seed'].unique():
        seed_rewards = df[(df['Model'] == model) & (df['seed'] == seed)].iloc[:, 2:].values.flatten()  
        rewards.append(ewma(seed_rewards, alpha))
    mean_curve = np.mean(rewards, axis=0)
    std_dev = np.std(rewards, axis=0)
    results[model] = (mean_curve, std_dev)
    plt.plot(iterations, mean_curve, label=f'{model}', marker='o', markevery=1, markersize=8)
    plt.fill_between(iterations, mean_curve - std_dev, mean_curve + std_dev, alpha=0.2)

plt.legend()
plt.xlabel('Predict Steps')
plt.ylabel('ADE')
# plt.title('EWMA of Model Rewards Across Seeds')
plt.savefig("sensitivity.png")
