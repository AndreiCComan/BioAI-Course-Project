# -*- coding: utf-8 -*-

from tqdm import tqdm
from deap import base
from deap import creator
from deap import tools

import neat

import numpy as np
import random
import multiprocessing
import gym
import logging
import pickle
logging.getLogger().setLevel(logging.INFO)

global logger
logger = logging.getLogger("performance_logger")
logger.addHandler(logging.FileHandler("MountainCar_v0_NEAT_performance.csv"))


"""
EA which uses NEAT to evolve an entire Neural-Network
"""
class NEATModel():

    def __init__(self, MU, NGEN, NCPU, gym_environment_name):
        self.MU = MU
        self.NGEN = NGEN
        self.NCPU = NCPU
        self.gym_environment_name = gym_environment_name
        self.environment = gym.make(self.gym_environment_name)

        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             "./mountainCar-neat.conf")

        self.model = ''

    def create_network(self, genome, config):
        return neat.nn.FeedForwardNetwork.create(genome, config)

    def evaluate_genome(self, genome, config):
        self.model = self.create_network(genome, config)

        individual_fitness = 0
        individual_has_finished = False
        individual_environment = self.environment.reset()
        while not individual_has_finished:
            #environment.render() # uncomment this if you want to render each candidate
            features = np.array(individual_environment[np.newaxis,...], dtype=float)[0]
            individual_next_action = np.argmax(self.model.activate(features))
            individual_environment, fitness, individual_has_finished, info = self.environment.step(individual_next_action)
            individual_fitness += fitness
        self.environment.close()

        return individual_fitness

    def evolve(self):

        logger.info("generation,best_fitness,mean_fitness,median_fitness")

        pop = neat.Population(self.config)
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        pop.add_reporter(neat.StdOutReporter(True))

        pe = neat.ParallelEvaluator(self.NCPU, self.evaluate_genome)
        winner = pop.run(pe.evaluate, self.NGEN)
        self.model = neat.nn.FeedForwardNetwork.create(winner, self.config)

        best_genomes = stats.most_fit_genomes
        fitness_mean = stats.get_fitness_mean()
        fitness_median = stats.get_fitness_median()
        for e in range(0, self.NGEN):
            logger.info("{},{},{},{}".format(e, best_genomes[e].fitness, fitness_mean[e], fitness_median[e]))


    def save(self):
        with open("./model_MountainCar_v0_NEAT.pt", 'wb') as f:
            pickle.dump(self.model, f)

    def load(self):
        with open("./model_MountainCar_v0_NEAT.pt", 'rb') as f:
            self.model = pickle.load(f)

    def play(self, numer_of_times):
        for _ in range(numer_of_times):
            individual_has_finished = False
            individual_environment = self.environment.reset()
            while not individual_has_finished:
                self.environment.render() # uncomment this if you want to render each candidate
                features = np.array(individual_environment[np.newaxis,...], dtype=float)[0]
                individual_next_action = np.argmax(self.model.activate(features))
                individual_environment, fitness, individual_has_finished, info = self.environment.step(individual_next_action)
            self.environment.close()

if __name__ == "__main__":

    MU = 50
    NGEN = 500
    NCPU = 4

    gym_environment_name = "MountainCar-v0"

    agent = NEATModel(MU, NGEN, NCPU, gym_environment_name)
    agent.evolve()
    agent.save()
    agent.load()
    agent.play(50)
