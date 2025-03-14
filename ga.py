# ga.py
"""
Genetic Algorithm module for Tanker Fleet Optimization.

This module defines:
  - The DEAP creator classes for multi-objective optimization.
  - Functions for creating individuals and populations.
  - Genetic operators (mutation, crossover, selection).
  - A local search routine for refining individuals.
  - The main run_ga() function that evolves the population.
"""

import random, copy
from deap import base, creator, tools
from deap.tools.emo import assignCrowdingDist
import streamlit as st
import config
from simulate import eval_individual

# -----------------------------------------------------------------------------
# Create DEAP Classes for Multi-objective Optimization.
# -----------------------------------------------------------------------------
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

def create_individual():
    """
    Create an individual solution representing a fleet plan.
    
    Each individual is a list with one decision per year.
    Each decision is a list:
      [buy_kc135, upgrade_kc135, retire_kc135,
       buy_kc46, upgrade_kc46, retire_kc46,
       buy_KC46B, upgrade_KC46B, retire_KC46B,
       usage_profile]
       
    The first year (year 0) has no actions.
    """
    individual = []
    for y in range(config.YEARS):
        if y == 0:
            buy_kc135 = 0; upgrade_kc135 = 0; retire_kc135 = 0
            buy_kc46 = 0; upgrade_kc46 = 0; retire_kc46 = 0
            buy_KC46B = 0; upgrade_KC46B = 0; retire_KC46B = 0
        else:
            buy_kc135 = random.randint(0, config.TANKER_DATA["KC135"]["max_production_per_year"])
            upgrade_kc135 = random.randint(0, 1)
            retire_kc135 = random.randint(0, 10)
            
            buy_kc46 = random.randint(0, config.TANKER_DATA["KC46"]["max_production_per_year"])
            upgrade_kc46 = random.randint(0, 1)
            retire_kc46 = random.randint(0, 5)
            
            buy_KC46B = random.randint(0, config.TANKER_DATA["KC46B"]["max_production_per_year"])
            upgrade_KC46B = random.randint(0, 1)
            retire_KC46B = random.randint(0, 3)
        usage_profile = random.randint(0, config.USAGE_MAX)
        decision = [buy_kc135, upgrade_kc135, retire_kc135,
                    buy_kc46, upgrade_kc46, retire_kc46,
                    buy_KC46B, upgrade_KC46B, retire_KC46B,
                    usage_profile]
        individual.append(decision)
    return individual

toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def nested_mutation(individual, mu=0, sigma=2, indpb=0.2):
    """
    Mutate an individual by applying Gaussian perturbations to each decision element.
    """
    for decision in individual:
        for i in range(len(decision)):
            if random.random() < indpb:
                decision[i] = int(round(decision[i] + random.gauss(mu, sigma)))
                if decision[i] < 0:
                    decision[i] = 0
    return individual,

toolbox.register("mutate", nested_mutation, mu=0, sigma=2, indpb=0.2)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", eval_individual)

def local_search(ind, attempts=3):
    """
    Perform a local search on an individual solution.
    
    Attempts to improve the individual by mutating one element in a random year.
    Returns the best mutated individual and its fitness.
    """
    best_fit = ind.fitness.values
    best_copy = creator.Individual(ind[:])
    for _ in range(attempts):
        mutant = creator.Individual(best_copy[:])
        y_idx = random.randrange(config.YEARS)
        d = list(mutant[y_idx])
        elem = random.randrange(len(d))
        d[elem] += random.randint(-1, 1)
        # Ensure decision limits.
        if elem in [0]:
            d[elem] = max(0, min(d[elem], config.TANKER_DATA["KC135"]["max_production_per_year"]))
        elif elem in [3]:
            d[elem] = max(0, min(d[elem], config.TANKER_DATA["KC46"]["max_production_per_year"]))
        elif elem in [6]:
            d[elem] = max(0, min(d[elem], config.TANKER_DATA["KC46B"]["max_production_per_year"]))
        elif elem in [1,4,7]:
            d[elem] = 0 if d[elem] < 1 else 1
        else:
            d[elem] = max(0, d[elem])
        mutant[y_idx] = d
        fit = toolbox.evaluate(mutant)
        if (fit[0] < best_fit[0]) and (fit[1] >= best_fit[1]) and (fit[2] >= best_fit[2]):
            best_fit = fit
            best_copy = mutant
    return best_copy, best_fit

def assign_fitness(ind):
    """
    Assign fitness to an individual solution.
    """
    return toolbox.evaluate(ind)

def run_ga(progress_bar=None):
    """
    Execute the genetic algorithm (GA) for a specified number of generations.
    
    If a progress_bar object is provided, updates it after each generation.
    
    Returns:
      - The final population.
      - The Pareto front (non-dominated solutions).
    """
    random.seed(42)
    pop = toolbox.population(n=config.DEFAULT_POP_SIZE)
    pop = list(map(assign_fitness, pop))
    assignCrowdingDist(pop)
    for gen in range(1, config.DEFAULT_NGEN + 1):
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < config.DEFAULT_CXPB:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values
        for mutant in offspring:
            if random.random() < config.DEFAULT_MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        invalids = [ind for ind in offspring if not ind.fitness.valid]
        if invalids:
            evaluated = list(map(assign_fitness, invalids))
            for ind, fit in zip(invalids, evaluated):
                ind.fitness.values = fit.fitness.values
        assignCrowdingDist(offspring)
        pop = toolbox.select(pop + offspring, config.DEFAULT_POP_SIZE)
        assignCrowdingDist(pop)
        # Update progress bar if provided.
        if progress_bar is not None:
            progress = int((gen / config.DEFAULT_NGEN) * 100)
            progress_bar.progress(progress)
        # Optional: print generation progress.
        # st.write(f"Generation {gen} complete.")
    pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    return pop, pareto_front
