"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import csv
import os
import matplotlib.pyplot as plt



class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#


class Visualization:

    def plot1(self, algorithm):
        rewards = np.array(algorithm.rewards)
        trials = np.arange(1, len(rewards) + 1)
        avg_rewards = np.cumsum(rewards) / trials

        name = algorithm.__class__.__name__

        plt.figure(figsize=(10, 5))
        plt.plot(trials, avg_rewards, label=name)
        plt.axhline(4, color="red", linestyle="--", label="Best Reward")
        plt.xlabel("Trials")
        plt.ylabel("Average Reward")
        plt.title(name + " Learning Curve")
        plt.legend()
        plt.grid()
        plt.savefig("results/" + name + "_linear.png")

        plt.figure(figsize=(10, 5))
        plt.plot(trials, avg_rewards, label=name)
        plt.axhline(4, color="red", linestyle="--", label="Best Reward")
        plt.xscale("log")
        plt.xlabel("Trials")
        plt.ylabel("Average Reward")
        plt.title(name + " Learning Curve (Log)")
        plt.legend()
        plt.grid()
        plt.savefig("results/" + name + "_log.png")

    def plot2(self, epsilon, thompson):
        e_rewards = np.cumsum(epsilon.rewards)
        t_rewards = np.cumsum(thompson.rewards)

        plt.figure(figsize=(10, 5))
        plt.plot(e_rewards, label="Epsilon Greedy")
        plt.plot(t_rewards, label="Thompson Sampling")
        plt.title("Cumulative Rewards")
        plt.xlabel("Trials")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid()
        plt.savefig("results/cumulative_rewards.png")

        e_regrets = np.cumsum(epsilon.regrets)
        t_regrets = np.cumsum(thompson.regrets)

        plt.figure(figsize=(10, 5))
        plt.plot(e_regrets, label="Epsilon Greedy")
        plt.plot(t_regrets, label="Thompson Sampling")
        plt.title("Cumulative Regrets")
        plt.xlabel("Trials")
        plt.ylabel("Total Regret")
        plt.legend()
        plt.grid()
        plt.savefig("results/cumulative_regrets.png")

class EpsilonGreedy(Bandit):
    #Epsilon Greedy algorithm with decreasing exploration.

    def __init__(self, p):
        self.p = p
        self.mean = 0
        self.n = 0

    def __repr__(self):
        return f"EpsilonGreedy(p={self.p}, mean={self.mean:.2f})"

    def pull(self):
        return np.random.normal(self.p, 1)

    def update(self, reward):
        self.n += 1
        self.mean += (reward - self.mean) / self.n

    def experiment(self):
        bandits = [EpsilonGreedy(p) for p in [1, 2, 3, 4]]

        self.rewards = []
        self.regrets = []
        self.selected_bandits = []

        for i in range(4):
            reward = bandits[i].pull()
            bandits[i].update(reward)

            self.rewards.append(reward)
            self.regrets.append(4 - bandits[i].p)
            self.selected_bandits.append(i + 1)

        for t in range(5, 20001):
            epsilon = 1 / t

            if np.random.random() < epsilon:
                chosen = np.random.randint(4)
            else:
                chosen = np.argmax([b.mean for b in bandits])

            reward = bandits[chosen].pull()
            bandits[chosen].update(reward)

            self.rewards.append(reward)
            self.regrets.append(4 - bandits[chosen].p)
            self.selected_bandits.append(chosen + 1)

        self.bandits = bandits

    def report(self):
        logger.info(f"Epsilon Greedy Total Reward: {sum(self.rewards):.2f}")
        logger.info(f"Epsilon Greedy Total Regret: {sum(self.regrets):.2f}")
        logger.info(f"Average Reward: {np.mean(self.rewards):.4f}")
        logger.info(f"Average Regret: {np.mean(self.regrets):.4f}")

        with open("results/epsilon_rewards.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Bandit", "Reward", "Algorithm"])

            for i in range(len(self.rewards)):
                writer.writerow([
                    self.selected_bandits[i],
                    self.rewards[i],
                    "Epsilon Greedy"])


class ThompsonSampling(Bandit):
    #Thompson Sampling with Gaussian rewards and known precision.

    def __init__(self, p):
        self.p = p
        self.mean = 0
        self.n = 0
        self.prior_mean = 0
        self.prior_precision = 1
        self.precision = 1

    def __repr__(self):
        return f"ThompsonSampling(p={self.p}, mean={self.mean:.2f})"

    def pull(self):
        return np.random.normal(self.p, 1)

    def update(self, reward):
        self.n += 1

        new_precision = self.prior_precision + self.precision

        new_mean = (
            self.prior_precision * self.prior_mean
            + self.precision * reward) / new_precision

        self.prior_mean = new_mean
        self.prior_precision = new_precision
        self.mean = new_mean

    def experiment(self):
        bandits = [ThompsonSampling(p) for p in [1, 2, 3, 4]]

        self.rewards = []
        self.regrets = []
        self.selected_bandits = []

        for t in range(20000):

            samples = []
            for b in bandits:
                sample = np.random.normal(
                    b.prior_mean,
                    1 / np.sqrt(b.prior_precision))
                samples.append(sample)

            chosen = np.argmax(samples)

            reward = bandits[chosen].pull()
            bandits[chosen].update(reward)

            self.rewards.append(reward)
            self.regrets.append(4 - bandits[chosen].p)
            self.selected_bandits.append(chosen + 1)

        self.bandits = bandits

    def report(self):
        logger.info(f"Thompson Sampling Total Reward: {sum(self.rewards):.2f}")
        logger.info(f"Thompson Sampling Total Regret: {sum(self.regrets):.2f}")
        logger.info(f"Average Reward: {np.mean(self.rewards):.4f}")
        logger.info(f"Average Regret: {np.mean(self.regrets):.4f}")

        with open("results/thompson_rewards.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Bandit", "Reward", "Algorithm"])

            for i in range(len(self.rewards)):
                writer.writerow([
                    self.selected_bandits[i],
                    self.rewards[i],
                    "Thompson Sampling"])


def comparison(epsilon, thompson):

    with open("results/rewards.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Bandit", "Reward", "Algorithm"])

        for i in range(20000):
            writer.writerow([
                epsilon.selected_bandits[i],
                epsilon.rewards[i],
                "Epsilon Greedy"])

            writer.writerow([
                thompson.selected_bandits[i],
                thompson.rewards[i],
                "Thompson Sampling"])

    logger.info("Comparison Results")
    logger.info(f"Epsilon Greedy Reward: {sum(epsilon.rewards):.2f}")
    logger.info(f"Thompson Sampling Reward: {sum(thompson.rewards):.2f}")
    logger.info(f"Epsilon Greedy Regret: {sum(epsilon.regrets):.2f}")
    logger.info(f"Thompson Sampling Regret: {sum(thompson.regrets):.2f}")

if __name__ == "__main__":

    np.random.seed(42)
    os.makedirs("results", exist_ok=True)

    epsilon = EpsilonGreedy(0)
    epsilon.experiment()
    epsilon.report()

    thompson = ThompsonSampling(0)
    thompson.experiment()
    thompson.report()

    visualization = Visualization()
    visualization.plot1(epsilon)
    visualization.plot1(thompson)
    visualization.plot2(epsilon, thompson)

    comparison(epsilon, thompson)
    plt.show()