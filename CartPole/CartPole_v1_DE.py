# -*- coding: utf-8 -*-

from tqdm import tqdm
from deap import base
from deap import creator
from deap import tools

import torch
import torch.nn as nn
import numpy as np
import random
import multiprocessing
import gym
import logging
logging.getLogger().setLevel(logging.INFO)


class TorchModel():
    def __init__(self, CR, F, MU, NGEN, NCPU, gym_environment_name):
        self.CR = CR
        self.F = F
        self.MU = MU
        self.NGEN = NGEN
        self.NCPU = NCPU
        self.gym_environment_name = gym_environment_name
        self.environment = gym.make(self.gym_environment_name)

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

        self.model = self.generate_model()
        self.model = self.model.double()

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self.generate_individual, creator.Individual, self.model)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("select", tools.selRandom, k=3)
        self.toolbox.register("evaluate", self.evaluate_individual, self.environment, self.model)

        self.pop = self.toolbox.population(n=self.MU)
        self.hof = tools.HallOfFame(1, similar=np.array_equal)

    def generate_model(self):
        model = nn.Sequential(nn.Linear(self.environment.observation_space.shape[0], 8),
                                nn.ReLU(),
                                nn.Linear(8, self.environment.action_space.n),
                                nn.LogSoftmax(dim=0))
        for parameter in model.parameters():
            parameter.requires_grad = False
        return model


    def generate_individual(self, individual, model):
        weights = []
        for parameter in model.parameters():
            if len(parameter.size()) == 1:
                parameter_dim = parameter.size()[0]
                weights.append(np.random.rand(parameter_dim) * np.sqrt(1 / (parameter_dim)))
            else:
                parameter_dim_0, parameter_dim_1 = parameter.size()
                weights.append(
                    np.random.rand(parameter_dim_0, parameter_dim_1) * np.sqrt(1 / (parameter_dim_0 + parameter_dim_1)))
        return individual(np.array(weights))

    def evaluate_individual(self, environment, model, individual):
        for parameter, numpy_array in zip(model.parameters(), individual):
            parameter.data = torch.from_numpy(numpy_array)

        individual_fitness = 0
        individual_has_finished = False
        individual_environment = environment.reset()
        while not individual_has_finished:
            #environment.render() # uncomment this if you want to render each candidate
            features = np.array(individual_environment[np.newaxis,...], dtype=float)[0]
            individual_next_action = torch.argmax(model(torch.tensor(features, dtype=torch.double))).item()
            individual_environment, fitness, individual_has_finished, info = environment.step(individual_next_action)
            individual_fitness += fitness
        environment.close()

        return individual_fitness,

    def differential_evolution(self, agent, population):
        a, b, c = self.toolbox.select(population)
        y = self.toolbox.clone(agent)
        index = random.randrange(len(agent))

        for i, value in enumerate(agent):
            if i == index or random.random() < self.CR:
                y[i] = a[i] + self.F * (b[i] - c[i])
        
        y.fitness.values = self.toolbox.evaluate(y)
        
        if y.fitness > agent.fitness:
            return y
        else:
            return agent

    def evolve(self):
        for generation in tqdm(range(self.NGEN), total=self.NGEN):
            agents = [(agent, self.pop) for agent in self.pop]
            with multiprocessing.Pool(processes=self.NCPU) as pool:
                self.pop = pool.starmap(self.differential_evolution, agents)
            self.hof.update(self.pop)

        best_individual = self.hof[0]
        for parameter, numpy_array in zip(self.model.parameters(), best_individual):
            parameter.data = torch.from_numpy(numpy_array)

    def save(self):
        torch.save(self.model.state_dict(), "model_{}_DE.pt".format(self.gym_environment_name))

    def load(self):
        self.model.load_state_dict(torch.load("model_{}_DE.pt".format(self.gym_environment_name)))

    def play(self, numer_of_times):
        for _ in range(numer_of_times):
            individual_has_finished = False
            individual_environment = self.environment.reset()
            while not individual_has_finished:
                self.environment.render() # uncomment this if you want to render each candidate
                features = np.array(individual_environment[np.newaxis,...], dtype=float)[0]
                individual_next_action = torch.argmax(self.model(torch.tensor(features, dtype=torch.double))).item()
                individual_environment, fitness, individual_has_finished, info = self.environment.step(individual_next_action)
            self.environment.close()


if __name__ == "__main__":

    CR = 0.25
    F = 1.0
    MU = 50
    NGEN = 500
    NCPU = 4

    gym_environment_name = "CartPole-v1"

    agent = TorchModel(CR, F, MU, NGEN, NCPU, gym_environment_name)
    #agent.play(500)
    agent.evolve()
    agent.save()
    agent.load()
    agent.play(50)
