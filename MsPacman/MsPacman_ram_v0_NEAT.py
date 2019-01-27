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
logging.getLogger().setLevel(logging.INFO)

global logger
logger = logging.getLogger("performance_logger")
logger.addHandler(logging.FileHandler("MsPacman_ram_v0_DE_performance.csv"))


"""
EA which uses NEAT to evolve an entire Neural-Network
"""
class NEATModel():

    def __init__(self, CR, F, MU, NGEN, NCPU, gym_environment_name):
        self.CR = CR
        self.F = F
        self.MU = MU
        self.NGEN = NGEN
        self.NCPU = NCPU
        self.gym_environment_name = gym_environment_name
        self.environment = gym.make(self.gym_environment_name)

        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             "./pacman.conf")

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
        with open("./model.pt", 'wb') as f:
            pickle.dump(self.model, f)

if __name__ == "__main__":

    CR = 0.25
    F = 1.0
    MU = 50
    NGEN = 500
    NCPU = 4

    gym_environment_name = "MsPacman-ram-v0"

    agent = NEATModel(CR, F, MU, NGEN, NCPU, gym_environment_name)
    agent.evolve()
    agent.save()
    agent.load()
    agent.play(50)