# HW2: Multi-Armed Bandit Algorithms

## Overview

This homework implements and compares two multi-armed bandit algorithms: Epsilon Greedy and Thompson Sampling.

The goal is to find the best-performing bandit among four options by maximizing the total reward and minimizing regret.

## Experiment Setup

The experiment uses the following settings:

* Number of bandits: 4
* Expected rewards: [1, 2, 3, 4]
* Number of trials: 20,000 for each algorithm
* Reward distribution: Gaussian (normal distribution)
* Standard deviation: 1
* Random seed: 42

Bandit 4 has the highest expected reward, making it the optimal choice.

## Algorithms

### 1. Epsilon Greedy

Epsilon Greedy balances exploration and exploitation.

* Exploration means randomly selecting a bandit to learn about its rewards.
* Exploitation means selecting the bandit with the highest estimated average reward.
* The exploration probability decreases over time using ε = 1/t.
* Each bandit is tested once at the beginning.

### 2. Thompson Sampling

Thompson Sampling uses Bayesian updating to estimate the expected reward of each bandit.

* Each bandit starts with a Gaussian prior distribution.
* The algorithm samples a value from each band’s posterior distribution.
* It selects the bandit with the highest sampled value.
* After receiving a reward, the posterior distribution is updated.

## Results

The following results were obtained from the simulation:

Metric	Epsilon Greedy	Thompson Sampling
Total Reward	79,860.53	79,910.77
Total Regret	15	2
Average Reward	3.9930	3.9955
Average Regret	0.0008	0.0001

Both algorithms achieved average rewards close to the optimal expected reward of 4.

Thompson Sampling performed slightly better in this experiment, achieving a higher total reward and lower regret.

## Visualizations

The results folder contains six graphs:

* EpsilonGreedy_linear.png: Epsilon Greedy learning curve.
* EpsilonGreedy_log.png: Epsilon Greedy learning curve using a logarithmic scale.
* ThompsonSampling_linear.png: Thompson Sampling learning curve.
* ThompsonSampling_log.png: Thompson Sampling learning curve using a logarithmic scale.
* cumulative_rewards.png: Comparison of cumulative rewards.
* cumulative_regrets.png: Comparison of cumulative regrets.

The learning curves show how the average reward changes over time. The comparison graphs show the differences in total rewards and regret between the algorithms.

## Output Files

The program generates three CSV files:

* epsilon_rewards.csv
* thompson_rewards.csv
* rewards.csv

Each file contains the following columns:

Bandit, Reward, Algorithm

The combined rewards.csv file contains results from both algorithms.

## How to Run

1. Install the required libraries:
    pip install -r requirements.txt
2. Run the Python file:
    python Bandit.py
3. The program will display the results and generate CSV files and graphs in the results folder.

## Bonus: Proposed Improvements

One possible improvement would be to repeat the simulation using different random seeds instead of evaluating the algorithms based on a single run.

This would help determine whether one algorithm consistently performs better.

Another improvement would be to experiment with different exploration rates for Epsilon Greedy and different prior precision values for Thompson Sampling.

## Conclusion

Both algorithms successfully identified the bandit with the highest expected reward.
In this simulation, Thompson Sampling achieved slightly better results than Epsilon Greedy. However, additional simulations would be useful for comparing their performance more reliably.
