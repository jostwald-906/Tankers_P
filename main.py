# main.py
import random, math, copy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from deap import base, creator, tools
from deap.tools.emo import assignCrowdingDist

# Import default configuration
import config

# -------------------------
# GA Setup
# -------------------------
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)
toolbox = base.Toolbox()

def create_individual():
    ind = []
    for y in range(config.YEARS):
        if y == 0:
            buy_kc135 = 0; upgrade_kc135 = 0; retire_kc135 = 0
            buy_kc46 = 0; upgrade_kc46 = 0; retire_kc46 = 0
            buy_ngas = 0; upgrade_ngas = 0; retire_ngas = 0
        else:
            buy_kc135    = random.randint(0, config.TANKER_DATA["KC135"]["max_production_per_year"])
            upgrade_kc135= random.randint(0, 1)
            retire_kc135 = random.randint(0, 10)
            
            buy_kc46     = random.randint(0, config.TANKER_DATA["KC46"]["max_production_per_year"])
            upgrade_kc46 = random.randint(0, 1)
            retire_kc46  = random.randint(0, 5)
            
            buy_ngas     = random.randint(0, config.TANKER_DATA["NGAS"]["max_production_per_year"])
            upgrade_ngas = random.randint(0, 1)
            retire_ngas  = random.randint(0, 3)
        usage_profile = random.randint(0, config.USAGE_MAX)
        decision = [buy_kc135, upgrade_kc135, retire_kc135,
                    buy_kc46, upgrade_kc46, retire_kc46,
                    buy_ngas, upgrade_ngas, retire_ngas,
                    usage_profile]
        ind.append(decision)
    return ind

toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def nested_mutation(individual, mu=0, sigma=2, indpb=0.2):
    for decision in individual:
        for i in range(len(decision)):
            if random.random() < indpb:
                decision[i] = int(round(decision[i] + random.gauss(mu, sigma)))
                if decision[i] < 0:
                    decision[i] = 0
    return individual,

toolbox.register("mutate", nested_mutation, mu=0, sigma=2, indpb=0.2)

def check_overhaul(aircraft, base_prob):
    prob = base_prob + 0.01 * (aircraft["age"] // 10)
    return random.random() < prob

def simulate_fleet(individual):
    """
    Simulate the 30-year plan.
    Return (total_cost, total_capacity, total_capability, year_data).
    """
    kc135_fleet = [{"age": 50, "downtime": 0.0,
                    "survivability": config.TANKER_DATA["KC135"]["cap_survivability"],
                    "connectivity": config.TANKER_DATA["KC135"]["cap_connectivity"],
                    "multi_mission": config.TANKER_DATA["KC135"]["cap_multi_mission"]}
                   for _ in range(config.START_KC135_COUNT)]
    kc46_fleet = [{"age": 5, "downtime": 0.0,
                   "survivability": config.TANKER_DATA["KC46"]["cap_survivability"],
                   "connectivity": config.TANKER_DATA["KC46"]["cap_connectivity"],
                   "multi_mission": config.TANKER_DATA["KC46"]["cap_multi_mission"]}
                  for _ in range(config.START_KC46_COUNT)]
    ngas_fleet = [{"age": 0, "downtime": 0.0,
                   "survivability": config.TANKER_DATA["NGAS"]["cap_survivability"],
                   "connectivity": config.TANKER_DATA["NGAS"]["cap_connectivity"],
                   "multi_mission": config.TANKER_DATA["NGAS"]["cap_multi_mission"]}
                  for _ in range(config.START_NGAS_COUNT)]
    
    used_budget_blocks = [0.0] * len(config.ROLLING_BUDGET)
    total_cost = total_capacity = total_capability = 0.0
    year_data = []

    for year, decision in enumerate(individual):
        block_idx = year // config.BLOCK_SIZE
        scenario = config.SCENARIOS[year]
        (buy_kc135, upgrade_kc135, retire_kc135,
         buy_kc46, upgrade_kc46, retire_kc46,
         buy_ngas, upgrade_ngas, retire_ngas,
         usage_profile) = decision
        
        buy_kc135 = min(buy_kc135, config.TANKER_DATA["KC135"]["max_production_per_year"])
        buy_kc46  = min(buy_kc46, config.TANKER_DATA["KC46"]["max_production_per_year"])
        buy_ngas  = min(buy_ngas, config.TANKER_DATA["NGAS"]["max_production_per_year"])
        if year < 10:
            buy_ngas = 0
        
        for _ in range(min(retire_kc135, len(kc135_fleet))):
            kc135_fleet.pop(0)
        for _ in range(min(retire_kc46, len(kc46_fleet))):
            kc46_fleet.pop(0)
        for _ in range(min(retire_ngas, len(ngas_fleet))):
            ngas_fleet.pop(0)
        
        for _ in range(buy_kc135):
            kc135_fleet.append({
                "age": 0, "downtime": 0.0,
                "survivability": config.TANKER_DATA["KC135"]["cap_survivability"],
                "connectivity": config.TANKER_DATA["KC135"]["cap_connectivity"],
                "multi_mission": config.TANKER_DATA["KC135"]["cap_multi_mission"]
            })
        for _ in range(buy_kc46):
            kc46_fleet.append({
                "age": 0, "downtime": 0.0,
                "survivability": config.TANKER_DATA["KC46"]["cap_survivability"],
                "connectivity": config.TANKER_DATA["KC46"]["cap_connectivity"],
                "multi_mission": config.TANKER_DATA["KC46"]["cap_multi_mission"]
            })
        for _ in range(buy_ngas):
            ngas_fleet.append({
                "age": 0, "downtime": 0.0,
                "survivability": config.TANKER_DATA["NGAS"]["cap_survivability"],
                "connectivity": config.TANKER_DATA["NGAS"]["cap_connectivity"],
                "multi_mission": config.TANKER_DATA["NGAS"]["cap_multi_mission"]
            })
        
        if upgrade_kc135 > 0 and kc135_fleet:
            used_budget_blocks[block_idx] += config.UPGRADES["KC135"]["cost"]
            kc135_fleet[0]["survivability"] += config.UPGRADES["KC135"]["delta_survivability"]
            kc135_fleet[0]["connectivity"] += config.UPGRADES["KC135"]["delta_connectivity"]
            kc135_fleet[0]["multi_mission"] += config.UPGRADES["KC135"]["delta_multi_mission"]
        if upgrade_kc46 > 0 and kc46_fleet:
            used_budget_blocks[block_idx] += config.UPGRADES["KC46"]["cost"]
            kc46_fleet[0]["survivability"] += config.UPGRADES["KC46"]["delta_survivability"]
            kc46_fleet[0]["connectivity"] += config.UPGRADES["KC46"]["delta_connectivity"]
            kc46_fleet[0]["multi_mission"] += config.UPGRADES["KC46"]["delta_multi_mission"]
        if upgrade_ngas > 0 and ngas_fleet:
            used_budget_blocks[block_idx] += config.UPGRADES["NGAS"]["cost"]
            ngas_fleet[0]["survivability"] += config.UPGRADES["NGAS"]["delta_survivability"]
            ngas_fleet[0]["connectivity"] += config.UPGRADES["NGAS"]["delta_connectivity"]
            ngas_fleet[0]["multi_mission"] += config.UPGRADES["NGAS"]["delta_multi_mission"]
        
        annual_cost = annual_capacity = annual_capability = 0.0
        maint_factor = config.SCENARIO_PARAMS[scenario]["maint_cost_factor"]
        demand = config.SCENARIO_PARAMS[scenario]["capacity_demand"]
        
        def process_fleet(fleet, type_key):
            nonlocal annual_cost, annual_capacity, annual_capability
            base_maint = config.TANKER_DATA[type_key]["maint_cost"]
            base_cap = config.TANKER_DATA[type_key]["capacity"]
            for ac in fleet:
                ac["age"] += 1
                if check_overhaul(ac, config.BASE_OVERHAUL_PROB[type_key]):
                    oh_cost = 2_500_000
                    if type_key == "KC135":
                        oh_cost = 2_000_000
                    elif type_key == "NGAS":
                        oh_cost = 3_000_000
                    used_budget_blocks[block_idx] += oh_cost
                    ac["downtime"] = 0.2
                usage_wear_cost = 500_000 if type_key != "NGAS" else 600_000
                m_cost = (base_maint + usage_wear_cost * usage_profile) * maint_factor
                annual_cost += m_cost
                downtime_frac = ac["downtime"]
                ac["downtime"] = 0.0
                annual_capacity += base_cap * (1.0 - downtime_frac)
                c_score = (ac["survivability"] * config.SUBCOMPONENT_WEIGHTS["survivability"] +
                           ac["connectivity"] * config.SUBCOMPONENT_WEIGHTS["connectivity"] +
                           ac["multi_mission"] * config.SUBCOMPONENT_WEIGHTS["multi_mission"])
                annual_capability += c_score * (1.0 - downtime_frac)
        
        process_fleet(kc135_fleet, "KC135")
        process_fleet(kc46_fleet, "KC46")
        process_fleet(ngas_fleet, "NGAS")
        
        proc_cost_kc135 = config.TANKER_DATA["KC135"]["proc_cost"] * buy_kc135
        proc_cost_kc46  = config.TANKER_DATA["KC46"]["proc_cost"] * buy_kc46
        proc_cost_ngas  = config.TANKER_DATA["NGAS"]["proc_cost"] * buy_ngas
        year_proc_cost = proc_cost_kc135 + proc_cost_kc46 + proc_cost_ngas
        used_budget_blocks[block_idx] += year_proc_cost
        annual_cost += year_proc_cost
        annual_cost = annual_cost / ((1 + config.DISCOUNT_RATE) ** year)
        total_cost += annual_cost
        
        fleet_size = len(kc135_fleet) + len(kc46_fleet) + len(ngas_fleet)
        if fleet_size < config.MIN_FLEET_SIZE:
            total_cost += 1e15
        if annual_capacity < demand:
            total_cost += 1e9
        total_capacity += annual_capacity
        total_capability += annual_capability
        
        year_data.append({
            "Year": 2025 + year,
            "KC135": len(kc135_fleet),
            "KC135_bought": buy_kc135,
            "KC135_retired": retire_kc135,
            "KC135_upgraded": upgrade_kc135,
            "KC46": len(kc46_fleet),
            "KC46_bought": buy_kc46,
            "KC46_retired": retire_kc46,
            "KC46_upgraded": upgrade_kc46,
            "NGAS": len(ngas_fleet),
            "NGAS_bought": buy_ngas,
            "NGAS_retired": retire_ngas,
            "NGAS_upgraded": upgrade_ngas
        })
    
    return (total_cost, total_capacity, total_capability, year_data)

def eval_individual(ind):
    cost, capacity, capability, year_data = simulate_fleet(ind)
    ind.fitness.values = (cost, capacity, capability)
    ind.year_data = copy.deepcopy(year_data)
    return ind

toolbox.register("evaluate", eval_individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("select", tools.selNSGA2)

def local_search(ind, attempts=3):
    best_fit = ind.fitness.values
    best_copy = creator.Individual(ind[:])
    for _ in range(attempts):
        mutant = creator.Individual(best_copy[:])
        y_idx = random.randrange(config.YEARS)
        d = list(mutant[y_idx])
        elem = random.randrange(len(d))
        d[elem] += random.randint(-1, 1)
        if elem in [0]:
            d[elem] = max(0, min(d[elem], config.TANKER_DATA["KC135"]["max_production_per_year"]))
        elif elem in [3]:
            d[elem] = max(0, min(d[elem], config.TANKER_DATA["KC46"]["max_production_per_year"]))
        elif elem in [6]:
            d[elem] = max(0, min(d[elem], config.TANKER_DATA["NGAS"]["max_production_per_year"]))
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
    return toolbox.evaluate(ind)

def run_ga():
    random.seed(42)
    pop = toolbox.population(config.DEFAULT_POP_SIZE)
    pop = list(toolbox.map(assign_fitness, pop))
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
            evaluated = list(toolbox.map(assign_fitness, invalids))
            for ind, fit in zip(invalids, evaluated):
                ind.fitness.values = fit.fitness.values
        assignCrowdingDist(offspring)
        pop = toolbox.select(pop + offspring, config.DEFAULT_POP_SIZE)
        assignCrowdingDist(pop)
        if gen % 5 == 0:
            st.write(f"Generation {gen} complete.")
    pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    return pop, pareto_front

# -------------------------
# Streamlit App for Interactive Visualization
# -------------------------
st.set_page_config(layout="wide")
st.title("Tanker Fleet Optimization with GA")

# --- Left Sidebar: GA Parameters ---
st.sidebar.header("GA Parameters")
pop_size_input = st.sidebar.number_input("Population Size", min_value=10, max_value=1000, value=config.DEFAULT_POP_SIZE)
n_gen_input = st.sidebar.number_input("Number of Generations", min_value=5, max_value=100, value=config.DEFAULT_NGEN)
cxpb_input = st.sidebar.slider("Crossover Probability", min_value=0.0, max_value=1.0, value=config.DEFAULT_CXPB)
mutpb_input = st.sidebar.slider("Mutation Probability", min_value=0.0, max_value=1.0, value=config.DEFAULT_MUTPB)
# (Removed the duplicate st.sidebar.button("Run GA") here)

# --- Right Panel: Model Assumptions ---
with st.expander("Model Assumptions (Click to Expand/Collapse)", expanded=True):
    new_years = st.number_input("Years", min_value=1, value=config.YEARS)
    new_block_size = st.number_input("Block Size", min_value=1, value=config.BLOCK_SIZE)
    new_min_fleet = st.number_input("Min Fleet Size", min_value=1, value=config.MIN_FLEET_SIZE)
    new_discount = st.number_input("Discount Rate", min_value=0.0, max_value=1.0, step=0.01, value=config.DISCOUNT_RATE)
    
    st.markdown("#### Scenario Parameters")
    new_peacetime_maint = st.number_input("Peacetime Maintenance Factor", value=config.SCENARIO_PARAMS["peacetime"]["maint_cost_factor"], step=0.1)
    new_peacetime_cap = st.number_input("Peacetime Capacity Demand", value=config.SCENARIO_PARAMS["peacetime"]["capacity_demand"], step=10)
    
    new_surge_maint = st.number_input("Surge Maintenance Factor", value=config.SCENARIO_PARAMS["surge"]["maint_cost_factor"], step=0.1)
    new_surge_cap = st.number_input("Surge Capacity Demand", value=config.SCENARIO_PARAMS["surge"]["capacity_demand"], step=10)
    
    new_high_maint = st.number_input("High Threat Maintenance Factor", value=config.SCENARIO_PARAMS["high_threat"]["maint_cost_factor"], step=0.1)
    new_high_cap = st.number_input("High Threat Capacity Demand", value=config.SCENARIO_PARAMS["high_threat"]["capacity_demand"], step=10)
    
    st.markdown("#### Tanker Data")
    st.markdown("##### KC135")
    new_kc135_capacity = st.number_input("KC135 Capacity", value=config.TANKER_DATA["KC135"]["capacity"], step=10)
    new_kc135_maint = st.number_input("KC135 Maintenance Cost", value=config.TANKER_DATA["KC135"]["maint_cost"], step=100000)
    new_kc135_proc = st.number_input("KC135 Procurement Cost", value=config.TANKER_DATA["KC135"]["proc_cost"], step=1000000)
    new_kc135_max = st.number_input("KC135 Max Production", value=config.TANKER_DATA["KC135"]["max_production_per_year"], step=1)
    new_kc135_surv = st.number_input("KC135 Survivability", value=config.TANKER_DATA["KC135"]["cap_survivability"], step=0.05, format="%.2f")
    new_kc135_conn = st.number_input("KC135 Connectivity", value=config.TANKER_DATA["KC135"]["cap_connectivity"], step=0.05, format="%.2f")
    new_kc135_mm = st.number_input("KC135 Multi-Mission", value=config.TANKER_DATA["KC135"]["cap_multi_mission"], step=0.05, format="%.2f")
    
    st.markdown("##### KC46")
    new_kc46_capacity = st.number_input("KC46 Capacity", value=config.TANKER_DATA["KC46"]["capacity"], step=10)
    new_kc46_maint = st.number_input("KC46 Maintenance Cost", value=config.TANKER_DATA["KC46"]["maint_cost"], step=100000)
    new_kc46_proc = st.number_input("KC46 Procurement Cost", value=config.TANKER_DATA["KC46"]["proc_cost"], step=1000000)
    new_kc46_max = st.number_input("KC46 Max Production", value=config.TANKER_DATA["KC46"]["max_production_per_year"], step=1)
    new_kc46_surv = st.number_input("KC46 Survivability", value=config.TANKER_DATA["KC46"]["cap_survivability"], step=0.05, format="%.2f")
    new_kc46_conn = st.number_input("KC46 Connectivity", value=config.TANKER_DATA["KC46"]["cap_connectivity"], step=0.05, format="%.2f")
    new_kc46_mm = st.number_input("KC46 Multi-Mission", value=config.TANKER_DATA["KC46"]["cap_multi_mission"], step=0.05, format="%.2f")
    
    st.markdown("##### NGAS")
    new_ngas_capacity = st.number_input("NGAS Capacity", value=config.TANKER_DATA["NGAS"]["capacity"], step=10)
    new_ngas_maint = st.number_input("NGAS Maintenance Cost", value=config.TANKER_DATA["NGAS"]["maint_cost"], step=100000)
    new_ngas_proc = st.number_input("NGAS Procurement Cost", value=config.TANKER_DATA["NGAS"]["proc_cost"], step=1000000)
    new_ngas_max = st.number_input("NGAS Max Production", value=config.TANKER_DATA["NGAS"]["max_production_per_year"], step=1)
    new_ngas_surv = st.number_input("NGAS Survivability", value=config.TANKER_DATA["NGAS"]["cap_survivability"], step=0.05, format="%.2f")
    new_ngas_conn = st.number_input("NGAS Connectivity", value=config.TANKER_DATA["NGAS"]["cap_connectivity"], step=0.05, format="%.2f")
    new_ngas_mm = st.number_input("NGAS Multi-Mission", value=config.TANKER_DATA["NGAS"]["cap_multi_mission"], step=0.05, format="%.2f")
    
    st.markdown("#### Upgrades")
    st.markdown("##### KC135")
    new_kc135_upg_cost = st.number_input("KC135 Upgrade Cost", value=config.UPGRADES["KC135"]["cost"], step=100000)
    new_kc135_delta_surv = st.number_input("KC135 Delta Survivability", value=config.UPGRADES["KC135"]["delta_survivability"], step=0.05, format="%.2f")
    new_kc135_delta_conn = st.number_input("KC135 Delta Connectivity", value=config.UPGRADES["KC135"]["delta_connectivity"], step=0.05, format="%.2f")
    new_kc135_delta_mm = st.number_input("KC135 Delta Multi-Mission", value=config.UPGRADES["KC135"]["delta_multi_mission"], step=0.05, format="%.2f")
    
    st.markdown("##### KC46")
    new_kc46_upg_cost = st.number_input("KC46 Upgrade Cost", value=config.UPGRADES["KC46"]["cost"], step=100000)
    new_kc46_delta_surv = st.number_input("KC46 Delta Survivability", value=config.UPGRADES["KC46"]["delta_survivability"], step=0.05, format="%.2f")
    new_kc46_delta_conn = st.number_input("KC46 Delta Connectivity", value=config.UPGRADES["KC46"]["delta_connectivity"], step=0.05, format="%.2f")
    new_kc46_delta_mm = st.number_input("KC46 Delta Multi-Mission", value=config.UPGRADES["KC46"]["delta_multi_mission"], step=0.05, format="%.2f")
    
    st.markdown("##### NGAS")
    new_ngas_upg_cost = st.number_input("NGAS Upgrade Cost", value=config.UPGRADES["NGAS"]["cost"], step=100000)
    new_ngas_delta_surv = st.number_input("NGAS Delta Survivability", value=config.UPGRADES["NGAS"]["delta_survivability"], step=0.05, format="%.2f")
    new_ngas_delta_conn = st.number_input("NGAS Delta Connectivity", value=config.UPGRADES["NGAS"]["delta_connectivity"], step=0.05, format="%.2f")
    new_ngas_delta_mm = st.number_input("NGAS Delta Multi-Mission", value=config.UPGRADES["NGAS"]["delta_multi_mission"], step=0.05, format="%.2f")
    
    if st.button("Update Model Assumptions", key="update_assumptions"):
        config.YEARS = int(new_years)
        config.BLOCK_SIZE = int(new_block_size)
        config.MIN_FLEET_SIZE = int(new_min_fleet)
        config.DISCOUNT_RATE = float(new_discount)
        
        config.SCENARIO_PARAMS["peacetime"]["maint_cost_factor"] = float(new_peacetime_maint)
        config.SCENARIO_PARAMS["peacetime"]["capacity_demand"] = int(new_peacetime_cap)
        config.SCENARIO_PARAMS["surge"]["maint_cost_factor"] = float(new_surge_maint)
        config.SCENARIO_PARAMS["surge"]["capacity_demand"] = int(new_surge_cap)
        config.SCENARIO_PARAMS["high_threat"]["maint_cost_factor"] = float(new_high_maint)
        config.SCENARIO_PARAMS["high_threat"]["capacity_demand"] = int(new_high_cap)
        
        config.TANKER_DATA["KC135"]["capacity"] = int(new_kc135_capacity)
        config.TANKER_DATA["KC135"]["maint_cost"] = int(new_kc135_maint)
        config.TANKER_DATA["KC135"]["proc_cost"] = int(new_kc135_proc)
        config.TANKER_DATA["KC135"]["max_production_per_year"] = int(new_kc135_max)
        config.TANKER_DATA["KC135"]["cap_survivability"] = float(new_kc135_surv)
        config.TANKER_DATA["KC135"]["cap_connectivity"] = float(new_kc135_conn)
        config.TANKER_DATA["KC135"]["cap_multi_mission"] = float(new_kc135_mm)
        
        config.TANKER_DATA["KC46"]["capacity"] = int(new_kc46_capacity)
        config.TANKER_DATA["KC46"]["maint_cost"] = int(new_kc46_maint)
        config.TANKER_DATA["KC46"]["proc_cost"] = int(new_kc46_proc)
        config.TANKER_DATA["KC46"]["max_production_per_year"] = int(new_kc46_max)
        config.TANKER_DATA["KC46"]["cap_survivability"] = float(new_kc46_surv)
        config.TANKER_DATA["KC46"]["cap_connectivity"] = float(new_kc46_conn)
        config.TANKER_DATA["KC46"]["cap_multi_mission"] = float(new_kc46_mm)
        
        config.TANKER_DATA["NGAS"]["capacity"] = int(new_ngas_capacity)
        config.TANKER_DATA["NGAS"]["maint_cost"] = int(new_ngas_maint)
        config.TANKER_DATA["NGAS"]["proc_cost"] = int(new_ngas_proc)
        config.TANKER_DATA["NGAS"]["max_production_per_year"] = int(new_ngas_max)
        config.TANKER_DATA["NGAS"]["cap_survivability"] = float(new_ngas_surv)
        config.TANKER_DATA["NGAS"]["cap_connectivity"] = float(new_ngas_conn)
        config.TANKER_DATA["NGAS"]["cap_multi_mission"] = float(new_ngas_mm)
        
        config.UPGRADES["KC135"]["cost"] = int(new_kc135_upg_cost)
        config.UPGRADES["KC135"]["delta_survivability"] = float(new_kc135_delta_surv)
        config.UPGRADES["KC135"]["delta_connectivity"] = float(new_kc135_delta_conn)
        config.UPGRADES["KC135"]["delta_multi_mission"] = float(new_kc135_delta_mm)
        
        config.UPGRADES["KC46"]["cost"] = int(new_kc46_upg_cost)
        config.UPGRADES["KC46"]["delta_survivability"] = float(new_kc46_delta_surv)
        config.UPGRADES["KC46"]["delta_connectivity"] = float(new_kc46_delta_conn)
        config.UPGRADES["KC46"]["delta_multi_mission"] = float(new_kc46_delta_mm)
        
        config.UPGRADES["NGAS"]["cost"] = int(new_ngas_upg_cost)
        config.UPGRADES["NGAS"]["delta_survivability"] = float(new_ngas_delta_surv)
        config.UPGRADES["NGAS"]["delta_connectivity"] = float(new_ngas_delta_conn)
        config.UPGRADES["NGAS"]["delta_multi_mission"] = float(new_ngas_delta_mm)
        
        st.success("Model assumptions updated.")

# --- Left Sidebar: GA Parameters and Run GA button ---
if st.sidebar.button("Run GA", key="run_ga_button"):
    st.write("Running GA. Please wait...")
    import multiprocessing
    from multiprocessing import freeze_support
    freeze_support()
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    
    config.DEFAULT_POP_SIZE = int(pop_size_input)
    config.DEFAULT_NGEN = int(n_gen_input)
    config.DEFAULT_CXPB = float(cxpb_input)
    config.DEFAULT_MUTPB = float(mutpb_input)
    
    pop, pareto = run_ga()
    costs = [s.fitness.values[0] for s in pareto]
    caps  = [s.fitness.values[1] for s in pareto]
    cabs  = [s.fitness.values[2] for s in pareto]
    st.session_state['pareto'] = pareto
    st.session_state['costs'] = costs
    st.session_state['caps'] = caps
    st.session_state['cabs'] = cabs
    
    timeseries_data = {}
    for i, sol in enumerate(pareto):
        df = pd.DataFrame(sol.year_data)
        df["Total"] = df["KC135"] + df["KC46"] + df["NGAS"]
        timeseries_data[i] = df
    st.session_state['timeseries_data'] = timeseries_data
    
    pool.close()
    pool.join()
    st.success("GA Completed.")

if 'pareto' in st.session_state:
    fig3d = go.Figure(data=[go.Scatter3d(
        x=st.session_state['costs'],
        y=st.session_state['caps'],
        z=st.session_state['cabs'],
        mode='markers',
        marker=dict(size=7, color=st.session_state['cabs'], colorscale='Viridis'),
        text=[f"Solution {i}" for i in range(len(st.session_state['costs']))],
        customdata=list(range(len(st.session_state['costs'])))
    )])
    fig3d.update_layout(
        title="3D Pareto Front",
        scene=dict(
            xaxis_title="Total Cost",
            yaxis_title="Total Capacity",
            zaxis_title="Total Capability"
        )
    )
    st.plotly_chart(fig3d, use_container_width=True)
    
    sol_index = st.selectbox("Select a Pareto Solution", list(range(len(st.session_state['costs']))))
    df_sol = st.session_state['timeseries_data'][sol_index]
    
    fig_area = go.Figure()
    fig_area.add_trace(go.Scatter(
        x=df_sol['Year'], y=df_sol['KC135'],
        mode='lines', name='KC135', stackgroup='one'
    ))
    fig_area.add_trace(go.Scatter(
        x=df_sol['Year'], y=df_sol['KC46'],
        mode='lines', name='KC46', stackgroup='one'
    ))
    fig_area.add_trace(go.Scatter(
        x=df_sol['Year'], y=df_sol['NGAS'],
        mode='lines', name='NGAS', stackgroup='one'
    ))
    fig_area.add_trace(go.Scatter(
        x=df_sol['Year'], y=df_sol['Total'],
        mode='lines+markers', name='Total Fleet',
        line=dict(color='black', width=2, dash='dot')
    ))
    fig_area.update_layout(
        title="Fleet Composition Over Time",
        xaxis_title="Year",
        yaxis_title="Fleet Count",
        hovermode='x unified'
    )
    st.plotly_chart(fig_area, use_container_width=True)
    
    required_cols = ["Year", "KC135_bought", "KC135_retired", "KC135_upgraded",
                     "KC46_bought", "KC46_retired", "KC46_upgraded",
                     "NGAS_bought", "NGAS_retired", "NGAS_upgraded", "Total"]
    for col in required_cols:
        if col not in df_sol.columns:
            df_sol[col] = 0
    st.subheader("Yearly Decision Details")
    st.table(df_sol[required_cols])
