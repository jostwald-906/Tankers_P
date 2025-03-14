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
    # Create an individual with decisions for each year (length = config.YEARS)
    individual = []
    for y in range(config.YEARS):
        if y == 0:
            buy_kc135 = 0; upgrade_kc135 = 0; retire_kc135 = 0
            buy_kc46  = 0; upgrade_kc46  = 0; retire_kc46  = 0
            buy_KC46B  = 0; upgrade_KC46B  = 0; retire_KC46B  = 0
        else:
            buy_kc135    = random.randint(0, config.TANKER_DATA["KC135"]["max_production_per_year"])
            upgrade_kc135= random.randint(0, 1)
            retire_kc135 = random.randint(0, 10)
            
            buy_kc46     = random.randint(0, config.TANKER_DATA["KC46"]["max_production_per_year"])
            upgrade_kc46 = random.randint(0, 1)
            retire_kc46  = random.randint(0, 5)
            
            buy_KC46B     = random.randint(0, config.TANKER_DATA["KC46B"]["max_production_per_year"])
            upgrade_KC46B = random.randint(0, 1)
            retire_KC46B  = random.randint(0, 3)
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
    Simulate the 30 (or updated number of) year plan.
    Returns (total_cost, total_capacity, total_capability, year_data).
    """
    # Build initial fleets using starting counts from config
    kc135_fleet = [{
        "age": 50, "downtime": 0.0,
        "survivability": config.TANKER_DATA["KC135"]["cap_survivability"],
        "connectivity": config.TANKER_DATA["KC135"]["cap_connectivity"],
        "multi_mission": config.TANKER_DATA["KC135"]["cap_multi_mission"]
    } for _ in range(config.START_KC135_COUNT)]
    
    kc46_fleet = [{
        "age": 5, "downtime": 0.0,
        "survivability": config.TANKER_DATA["KC46"]["cap_survivability"],
        "connectivity": config.TANKER_DATA["KC46"]["cap_connectivity"],
        "multi_mission": config.TANKER_DATA["KC46"]["cap_multi_mission"]
    } for _ in range(config.START_KC46_COUNT)]
    
    KC46B_fleet = [{
        "age": 0, "downtime": 0.0,
        "survivability": config.TANKER_DATA["KC46B"]["cap_survivability"],
        "connectivity": config.TANKER_DATA["KC46B"]["cap_connectivity"],
        "multi_mission": config.TANKER_DATA["KC46B"]["cap_multi_mission"]
    } for _ in range(config.START_KC46B_COUNT)]
    
    # Recompute SCENARIOS if YEARS changed:
    config.SCENARIOS = []
    for y in range(config.YEARS):
        if y < 5:
            config.SCENARIOS.append("peacetime")
        elif y < 10:
            config.SCENARIOS.append("surge")
        elif y < 20:
            config.SCENARIOS.append("high_threat")
        else:
            config.SCENARIOS.append("peacetime")
    
    # Adjust ROLLING_BUDGET to match number of blocks
    num_blocks = math.ceil(config.YEARS / config.BLOCK_SIZE)
    if len(config.ROLLING_BUDGET) != num_blocks:
        config.ROLLING_BUDGET = (config.ROLLING_BUDGET * num_blocks)[:num_blocks]
    
    used_budget_blocks = [0.0] * len(config.ROLLING_BUDGET)
    total_cost = total_capacity = total_capability = 0.0
    year_data = []

    for year, decision in enumerate(individual):
        block_idx = year // config.BLOCK_SIZE
        scenario = config.SCENARIOS[year]
        (buy_kc135, upgrade_kc135, retire_kc135,
         buy_kc46, upgrade_kc46, retire_kc46,
         buy_KC46B, upgrade_KC46B, retire_KC46B,
         usage_profile) = decision
        
        buy_kc135 = min(buy_kc135, config.TANKER_DATA["KC135"]["max_production_per_year"])
        buy_kc46  = min(buy_kc46, config.TANKER_DATA["KC46"]["max_production_per_year"])
        buy_KC46B  = min(buy_KC46B, config.TANKER_DATA["KC46B"]["max_production_per_year"])
        if year < 10:
            buy_KC46B = 0
        
        # Store initial counts before retirement actions:
        initial_kc135 = len(kc135_fleet)
        initial_kc46  = len(kc46_fleet)
        initial_KC46B = len(KC46B_fleet)
        
        # Calculate effective retirements
        effective_retire_kc135 = min(retire_kc135, initial_kc135)
        effective_retire_kc46  = min(retire_kc46, initial_kc46)
        effective_retire_KC46B = min(retire_KC46B, initial_KC46B)
        
        # Process retirements using the effective numbers:
        for _ in range(effective_retire_kc135):
            kc135_fleet.pop(0)
        for _ in range(effective_retire_kc46):
            kc46_fleet.pop(0)
        for _ in range(effective_retire_KC46B):
            KC46B_fleet.pop(0)
        
        # Process buy actions (these are always applied as is)
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
        for _ in range(buy_KC46B):
            KC46B_fleet.append({
                "age": 0, "downtime": 0.0,
                "survivability": config.TANKER_DATA["KC46B"]["cap_survivability"],
                "connectivity": config.TANKER_DATA["KC46B"]["cap_connectivity"],
                "multi_mission": config.TANKER_DATA["KC46B"]["cap_multi_mission"]
            })
        
        # Compute effective upgrades.
        # Upgrades are only applied if there's at least one aircraft after retire and buy actions.
        effective_upgrade_kc135 = 1 if (upgrade_kc135 > 0 and len(kc135_fleet) > 0) else 0
        effective_upgrade_kc46  = 1 if (upgrade_kc46 > 0 and len(kc46_fleet) > 0) else 0
        effective_upgrade_KC46B = 1 if (upgrade_KC46B > 0 and len(KC46B_fleet) > 0) else 0
        
        # Apply upgrades only if effective upgrade is 1
        if effective_upgrade_kc135:
            used_budget_blocks[block_idx] += config.UPGRADES["KC135"]["cost"]
            kc135_fleet[0]["survivability"] += config.UPGRADES["KC135"]["delta_survivability"]
            kc135_fleet[0]["connectivity"] += config.UPGRADES["KC135"]["delta_connectivity"]
            kc135_fleet[0]["multi_mission"] += config.UPGRADES["KC135"]["delta_multi_mission"]
        if effective_upgrade_kc46:
            used_budget_blocks[block_idx] += config.UPGRADES["KC46"]["cost"]
            kc46_fleet[0]["survivability"] += config.UPGRADES["KC46"]["delta_survivability"]
            kc46_fleet[0]["connectivity"] += config.UPGRADES["KC46"]["delta_connectivity"]
            kc46_fleet[0]["multi_mission"] += config.UPGRADES["KC46"]["delta_multi_mission"]
        if effective_upgrade_KC46B:
            used_budget_blocks[block_idx] += config.UPGRADES["KC46B"]["cost"]
            KC46B_fleet[0]["survivability"] += config.UPGRADES["KC46B"]["delta_survivability"]
            KC46B_fleet[0]["connectivity"] += config.UPGRADES["KC46B"]["delta_connectivity"]
            KC46B_fleet[0]["multi_mission"] += config.UPGRADES["KC46B"]["delta_multi_mission"]
        
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
                    elif type_key == "KC46B":
                        oh_cost = 3_000_000
                    used_budget_blocks[block_idx] += oh_cost
                    ac["downtime"] = 0.2
                usage_wear_cost = 500_000 if type_key != "KC46B" else 600_000
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
        process_fleet(KC46B_fleet, "KC46B")
        
        proc_cost_kc135 = config.TANKER_DATA["KC135"]["proc_cost"] * buy_kc135
        proc_cost_kc46  = config.TANKER_DATA["KC46"]["proc_cost"] * buy_kc46
        proc_cost_KC46B  = config.TANKER_DATA["KC46B"]["proc_cost"] * buy_KC46B
        year_proc_cost = proc_cost_kc135 + proc_cost_kc46 + proc_cost_KC46B
        used_budget_blocks[block_idx] += year_proc_cost
        annual_cost += year_proc_cost
        annual_cost = annual_cost / ((1 + config.DISCOUNT_RATE) ** year)
        total_cost += annual_cost
        
        fleet_size = len(kc135_fleet) + len(kc46_fleet) + len(KC46B_fleet)
        if fleet_size < config.MIN_FLEET_SIZE:
            total_cost += 1e15
        if annual_capacity < demand:
            total_cost += 1e9
        total_capacity += annual_capacity
        total_capability += annual_capability
        
        # (After computing annual_cost, annual_capacity, annual_capability)
        year_data.append({
            "Year": 2025 + year,
            "KC135": len(kc135_fleet),
            "KC135_bought": buy_kc135,
            "KC135_retired": effective_retire_kc135,
            "KC135_upgraded": effective_upgrade_kc135,
            "KC46": len(kc46_fleet),
            "KC46_bought": buy_kc46,
            "KC46_retired": effective_retire_kc46,
            "KC46_upgraded": effective_upgrade_kc46,
            "KC46B": len(KC46B_fleet),
            "KC46B_bought": buy_KC46B,
            "KC46B_retired": effective_retire_KC46B,
            "KC46B_upgraded": effective_upgrade_KC46B,
            "Annual_Cost": annual_cost,
            "Annual_Capacity": annual_capacity,
            "Annual_Capability": annual_capability
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

# Create two tabs: one for Model Info and one for running the model
tabs = st.tabs(["Model Info", "Run GA"])

with tabs[0]:
    st.header("About the Model")
    st.markdown("""
    **Inputs:**
    
    - **Simulation Parameters:** Number of simulation years, block size for budgeting, minimum fleet size requirement, discount rate, starting fleet counts, and rolling budgets.
    - **Scenario Parameters:** Maintenance cost multipliers and capacity demand requirements for peacetime, surge, and high‑threat scenarios.
    - **Tanker Data:** Data for KC‑135, KC‑46, and KC46B including refueling capacity, annual maintenance cost, procurement cost, maximum production per year, and baseline capability metrics (survivability, connectivity, multi‑mission).
    - **Upgrades:** Upgrade cost and improvement values (delta in survivability, connectivity, and multi‑mission capability) for each tanker type.
    
    **Approach:**
    
    - The model uses a multi‑objective Genetic Algorithm (GA) to optimize three objectives simultaneously:
      - **Cost:** Total cost is computed as the discounted sum of annual maintenance, procurement, and overhaul expenses. Penalties are added if the fleet fails to meet the minimum size or if annual capacity falls below demand.
      - **Capacity:** Each year’s effective refueling capacity is calculated by adjusting the base capacity of each aircraft by its downtime. This value approximates the aircraft’s average daily capacity.
      - **Capability:** A composite score is derived for each aircraft using a weighted sum of survivability, connectivity, and multi‑mission capability.
    - Annual decisions include buying, retiring, and upgrading aircraft (along with setting a usage profile), all within production limits. Spending is tracked against rolling budgets for each 5‑year block, though these budgets are recorded rather than enforced as hard constraints.
    - Scenario‑specific adjustments modify maintenance costs and capacity requirements according to whether the simulation period is peacetime, surge, or high‑threat.
    - The GA employs the NSGA‑II algorithm to explore trade‑offs between minimizing cost and maximizing capacity and capability, generating a Pareto front of optimal solutions.
    
    **Outputs:**
    
    - A 3D Pareto front visualizing the trade‑offs among total cost, capacity, and capability.
    - Stacked area charts that display how the fleet composition evolves over time.
    - A detailed table showing year‑by‑year decisions (purchases, retirements, upgrades) along with computed annual metrics.
    - An aggregated table that groups results into 5‑year periods (with the current year, e.g. 2025, as the first row), summarizing total cost, allocated budget, average annual capacity and capability, average counts for each aircraft type, and overall fleet size.
    """)

    
with tabs[1]:
    # --- Left Sidebar: GA Parameters ---
    st.sidebar.header("GA Parameters")
    pop_size_input = st.sidebar.number_input("Population Size", min_value=10, max_value=1000, value=config.DEFAULT_POP_SIZE,
                                               help="Number of individuals in the GA population.")
    n_gen_input = st.sidebar.number_input("Number of Generations", min_value=5, max_value=100, value=config.DEFAULT_NGEN,
                                          help="Number of generations for the GA.")
    cxpb_input = st.sidebar.slider("Crossover Probability", min_value=0.0, max_value=1.0, value=config.DEFAULT_CXPB,
                                   help="Probability that two individuals will crossover.")
    mutpb_input = st.sidebar.slider("Mutation Probability", min_value=0.0, max_value=1.0, value=config.DEFAULT_MUTPB,
                                    help="Probability that an individual will mutate.")
   # run_ga_button = st.sidebar.button("Run GA", key="run_ga_button")
    
    # --- Right Panel: Model Assumptions in an Expander ---
    with st.expander("Model Assumptions (Click to Expand/Collapse)", expanded=True):
        # General simulation parameters
        new_min_fleet = st.number_input("Min Fleet Size", min_value=1, value=config.MIN_FLEET_SIZE,
                                        help="Minimum required fleet size.")
        new_discount = st.number_input("Discount Rate", min_value=0.0, max_value=1.0, step=0.01, value=config.DISCOUNT_RATE,
                                       help="Rate used to discount future costs.")
        
        # Scenario parameters
        st.markdown("#### Scenario Parameters")
        new_peacetime_maint = st.number_input("Peacetime Maintenance Factor", value=config.SCENARIO_PARAMS["peacetime"]["maint_cost_factor"], step=0.1,
                                              help="Multiplier for maintenance costs in peacetime.")
        new_peacetime_cap = st.number_input("Peacetime Capacity Demand", value=config.SCENARIO_PARAMS["peacetime"]["capacity_demand"], step=10,
                                            help="Required capacity in peacetime.")
        new_surge_maint = st.number_input("Surge Maintenance Factor", value=config.SCENARIO_PARAMS["surge"]["maint_cost_factor"], step=0.1,
                                          help="Maintenance cost multiplier during surge.")
        new_surge_cap = st.number_input("Surge Capacity Demand", value=config.SCENARIO_PARAMS["surge"]["capacity_demand"], step=10,
                                        help="Capacity demand during surge.")
        new_high_maint = st.number_input("High Threat Maintenance Factor", value=config.SCENARIO_PARAMS["high_threat"]["maint_cost_factor"], step=0.1,
                                         help="Maintenance cost multiplier under high threat.")
        new_high_cap = st.number_input("High Threat Capacity Demand", value=config.SCENARIO_PARAMS["high_threat"]["capacity_demand"], step=10,
                                       help="Capacity demand under high threat.")
        
        # Tanker Data
        st.markdown("#### Tanker Data")
        st.markdown("##### KC135")
        new_kc135_capacity = st.number_input("KC135 Capacity", min_value=1, value=config.TANKER_DATA["KC135"]["capacity"], step=10,
                                             help="Refueling capacity of a KC-135.")
        new_kc135_maint = st.number_input("KC135 Maintenance Cost", min_value=1, value=config.TANKER_DATA["KC135"]["maint_cost"], step=100000,
                                          help="Annual maintenance cost for a KC-135.")
        new_kc135_proc = st.number_input("KC135 Procurement Cost", min_value=1, value=config.TANKER_DATA["KC135"]["proc_cost"], step=1000000,
                                         help="Cost to procure a new or refurbished KC-135.")
        new_kc135_max = st.number_input("KC135 Max Production", min_value=0, value=config.TANKER_DATA["KC135"]["max_production_per_year"], step=1,
                                        help="Maximum new KC-135s produced per year.")
        new_kc135_surv = st.number_input("KC135 Survivability", min_value=0.0, value=config.TANKER_DATA["KC135"]["cap_survivability"], step=0.05, format="%.2f",
                                         help="Baseline survivability score for a KC-135.")
        new_kc135_conn = st.number_input("KC135 Connectivity", min_value=0.0, value=config.TANKER_DATA["KC135"]["cap_connectivity"], step=0.05, format="%.2f",
                                         help="Baseline connectivity score for a KC-135.")
        new_kc135_mm = st.number_input("KC135 Multi-Mission", min_value=0.0, value=config.TANKER_DATA["KC135"]["cap_multi_mission"], step=0.05, format="%.2f",
                                       help="Baseline multi-mission capability for a KC-135.")
        
        st.markdown("##### KC46")
        new_kc46_capacity = st.number_input("KC46 Capacity", min_value=1, value=config.TANKER_DATA["KC46"]["capacity"], step=10,
                                            help="Refueling capacity of a KC-46.")
        new_kc46_maint = st.number_input("KC46 Maintenance Cost", min_value=1, value=config.TANKER_DATA["KC46"]["maint_cost"], step=100000,
                                         help="Annual maintenance cost for a KC-46.")
        new_kc46_proc = st.number_input("KC46 Procurement Cost", min_value=1, value=config.TANKER_DATA["KC46"]["proc_cost"], step=1000000,
                                        help="Cost to procure a new KC-46.")
        new_kc46_max = st.number_input("KC46 Max Production", min_value=0, value=config.TANKER_DATA["KC46"]["max_production_per_year"], step=1,
                                       help="Maximum new KC-46s produced per year.")
        new_kc46_surv = st.number_input("KC46 Survivability", min_value=0.0, value=config.TANKER_DATA["KC46"]["cap_survivability"], step=0.05, format="%.2f",
                                        help="Baseline survivability score for a KC-46.")
        new_kc46_conn = st.number_input("KC46 Connectivity", min_value=0.0, value=config.TANKER_DATA["KC46"]["cap_connectivity"], step=0.05, format="%.2f",
                                        help="Baseline connectivity score for a KC-46.")
        new_kc46_mm = st.number_input("KC46 Multi-Mission", min_value=0.0, value=config.TANKER_DATA["KC46"]["cap_multi_mission"], step=0.05, format="%.2f",
                                      help="Baseline multi-mission capability for a KC-46.")
        
        st.markdown("##### KC46B")
        new_KC46B_capacity = st.number_input("KC46B Capacity", min_value=1, value=config.TANKER_DATA["KC46B"]["capacity"], step=10,
                                            help="Refueling capacity of an KC46B aircraft.")
        new_KC46B_maint = st.number_input("KC46B Maintenance Cost", min_value=1, value=config.TANKER_DATA["KC46B"]["maint_cost"], step=100000,
                                         help="Annual maintenance cost for KC46B.")
        new_KC46B_proc = st.number_input("KC46B Procurement Cost", min_value=1, value=config.TANKER_DATA["KC46B"]["proc_cost"], step=1000000,
                                        help="Cost to procure an KC46B aircraft.")
        new_KC46B_max = st.number_input("KC46B Max Production", min_value=0, value=config.TANKER_DATA["KC46B"]["max_production_per_year"], step=1,
                                       help="Maximum new KC46B produced per year.")
        new_KC46B_surv = st.number_input("KC46B Survivability", min_value=0.0, value=config.TANKER_DATA["KC46B"]["cap_survivability"], step=0.05, format="%.2f",
                                        help="Baseline survivability score for KC46B.")
        new_KC46B_conn = st.number_input("KC46B Connectivity", min_value=0.0, value=config.TANKER_DATA["KC46B"]["cap_connectivity"], step=0.05, format="%.2f",
                                        help="Baseline connectivity score for KC46B.")
        new_KC46B_mm = st.number_input("KC46B Multi-Mission", min_value=0.0, value=config.TANKER_DATA["KC46B"]["cap_multi_mission"], step=0.05, format="%.2f",
                                      help="Baseline multi-mission capability for KC46B.")
        
        st.markdown("#### Upgrades")
        st.markdown("##### KC135")
        new_kc135_upg_cost = st.number_input("KC135 Upgrade Cost", min_value=1, value=config.UPGRADES["KC135"]["cost"], step=100000,
                                             help="Cost to upgrade a KC-135.")
        new_kc135_delta_surv = st.number_input("KC135 Delta Survivability", min_value=0.0, value=config.UPGRADES["KC135"]["delta_survivability"], step=0.05, format="%.2f",
                                               help="Increase in survivability after upgrade for a KC-135.")
        new_kc135_delta_conn = st.number_input("KC135 Delta Connectivity", min_value=0.0, value=config.UPGRADES["KC135"]["delta_connectivity"], step=0.05, format="%.2f",
                                               help="Increase in connectivity after upgrade for a KC-135.")
        new_kc135_delta_mm = st.number_input("KC135 Delta Multi-Mission", min_value=0.0, value=config.UPGRADES["KC135"]["delta_multi_mission"], step=0.05, format="%.2f",
                                             help="Increase in multi-mission capability for a KC-135 after upgrade.")
        
        st.markdown("##### KC46")
        new_kc46_upg_cost = st.number_input("KC46 Upgrade Cost", min_value=1, value=config.UPGRADES["KC46"]["cost"], step=100000,
                                            help="Cost to upgrade a KC-46.")
        new_kc46_delta_surv = st.number_input("KC46 Delta Survivability", min_value=0.0, value=config.UPGRADES["KC46"]["delta_survivability"], step=0.05, format="%.2f",
                                              help="Increase in survivability for a KC-46 after upgrade.")
        new_kc46_delta_conn = st.number_input("KC46 Delta Connectivity", min_value=0.0, value=config.UPGRADES["KC46"]["delta_connectivity"], step=0.05, format="%.2f",
                                              help="Increase in connectivity for a KC-46 after upgrade.")
        new_kc46_delta_mm = st.number_input("KC46 Delta Multi-Mission", min_value=0.0, value=config.UPGRADES["KC46"]["delta_multi_mission"], step=0.05, format="%.2f",
                                            help="Increase in multi-mission capability for a KC-46 after upgrade.")
        
        st.markdown("##### KC46B")
        new_KC46B_upg_cost = st.number_input("KC46B Upgrade Cost", min_value=1, value=config.UPGRADES["KC46B"]["cost"], step=100000,
                                            help="Cost to upgrade an KC46B aircraft.")
        new_KC46B_delta_surv = st.number_input("KC46B Delta Survivability", min_value=0.0, value=config.UPGRADES["KC46B"]["delta_survivability"], step=0.05, format="%.2f",
                                              help="Increase in survivability for KC46B after upgrade.")
        new_KC46B_delta_conn = st.number_input("KC46B Delta Connectivity", min_value=0.0, value=config.UPGRADES["KC46B"]["delta_connectivity"], step=0.05, format="%.2f",
                                              help="Increase in connectivity for KC46B after upgrade.")
        new_KC46B_delta_mm = st.number_input("KC46B Delta Multi-Mission", min_value=0.0, value=config.UPGRADES["KC46B"]["delta_multi_mission"], step=0.05, format="%.2f",
                                            help="Increase in multi-mission capability for KC46B after upgrade.")
        
        st.markdown("#### Starting Fleet & Budgets")
        new_start_kc135 = st.number_input("Starting KC-135 Count", min_value=0, value=config.START_KC135_COUNT, step=1,
                                          help="Initial number of KC-135 aircraft.")
        new_start_kc46 = st.number_input("Starting KC-46 Count", min_value=0, value=config.START_KC46_COUNT, step=1,
                                         help="Initial number of KC-46 aircraft.")
        new_start_KC46B = st.number_input("Starting KC46B Count", min_value=0, value=config.START_KC46B_COUNT, step=1,
                                         help="Initial number of KC46B aircraft.")

        if st.button("Update Model Assumptions", key="update_assumptions"):
            config.MIN_FLEET_SIZE = int(new_min_fleet)
            config.DISCOUNT_RATE = float(new_discount)
            
            # Recompute SCENARIOS based on new YEARS
            config.SCENARIOS = []
            for y in range(config.YEARS):
                if y < 5:
                    config.SCENARIOS.append("peacetime")
                elif y < 10:
                    config.SCENARIOS.append("surge")
                elif y < 20:
                    config.SCENARIOS.append("high_threat")
                else:
                    config.SCENARIOS.append("peacetime")
            
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
            
            config.TANKER_DATA["KC46B"]["capacity"] = int(new_KC46B_capacity)
            config.TANKER_DATA["KC46B"]["maint_cost"] = int(new_KC46B_maint)
            config.TANKER_DATA["KC46B"]["proc_cost"] = int(new_KC46B_proc)
            config.TANKER_DATA["KC46B"]["max_production_per_year"] = int(new_KC46B_max)
            config.TANKER_DATA["KC46B"]["cap_survivability"] = float(new_KC46B_surv)
            config.TANKER_DATA["KC46B"]["cap_connectivity"] = float(new_KC46B_conn)
            config.TANKER_DATA["KC46B"]["cap_multi_mission"] = float(new_KC46B_mm)
            
            config.UPGRADES["KC135"]["cost"] = int(new_kc135_upg_cost)
            config.UPGRADES["KC135"]["delta_survivability"] = float(new_kc135_delta_surv)
            config.UPGRADES["KC135"]["delta_connectivity"] = float(new_kc135_delta_conn)
            config.UPGRADES["KC135"]["delta_multi_mission"] = float(new_kc135_delta_mm)
            
            config.UPGRADES["KC46"]["cost"] = int(new_kc46_upg_cost)
            config.UPGRADES["KC46"]["delta_survivability"] = float(new_kc46_delta_surv)
            config.UPGRADES["KC46"]["delta_connectivity"] = float(new_kc46_delta_conn)
            config.UPGRADES["KC46"]["delta_multi_mission"] = float(new_kc46_delta_mm)
            
            config.UPGRADES["KC46B"]["cost"] = int(new_KC46B_upg_cost)
            config.UPGRADES["KC46B"]["delta_survivability"] = float(new_KC46B_delta_surv)
            config.UPGRADES["KC46B"]["delta_connectivity"] = float(new_KC46B_delta_conn)
            config.UPGRADES["KC46B"]["delta_multi_mission"] = float(new_KC46B_delta_mm)
            
            config.START_KC135_COUNT = int(new_start_kc135)
            config.START_KC46_COUNT = int(new_start_kc46)
            config.START_KC46B_COUNT = int(new_start_KC46B)
            
            # Update Rolling Budgets (comma-separated)
           
            # Clear any existing GA results from session state
            for key in ['pareto', 'costs', 'caps', 'cabs', 'timeseries_data']:
                if key in st.session_state:
                    del st.session_state[key]
            
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
            df["Total"] = df["KC135"] + df["KC46"] + df["KC46B"]
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
    
    
    # Number of frames and angles for a full 360° rotation.
    num_frames = 120
    angles = np.linspace(0, 360, num_frames)
    
    frames = []
    for angle in angles:
        # Rotate around the z-axis (capability axis) by adjusting the x and y of the camera.
        camera = dict(
            eye=dict(
                x=1.5 * np.cos(np.radians(angle)),
                y=1.5 * np.sin(np.radians(angle)),
                z=2  # keep z fixed so you're rotating around the capability axis
            )
        )
        frames.append(go.Frame(layout=dict(scene_camera=camera)))
    
    fig3d.frames = frames
    
    # Add an update menu with a play button to trigger the rotation.
    fig3d.update_layout(
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "buttons": [{
                "label": "Rotate",
                "method": "animate",
                "args": [None, {"frame": {"duration": 100, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0}}]
            }]
        }]
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
        x=df_sol['Year'], y=df_sol['KC46B'],
        mode='lines', name='KC46B', stackgroup='one'
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
                     "KC46B_bought", "KC46B_retired", "KC46B_upgraded", "Total"]
    for col in required_cols:
        if col not in df_sol.columns:
            df_sol[col] = 0
    st.subheader("Yearly Decision Details")
    st.table(df_sol[required_cols])
    
    
    #--- Attempt at new table
    # Assume df_sol is the DataFrame created from your solution’s year_data.
    #df_sol = st.session_state['timeseries_data'][sol_index].copy()
    # Get the current year (2025) data row from df_sol.
    current_year_data = df_sol[df_sol["Year"] == 2025].iloc[0]
    
    # Create a dictionary with the metrics for 2025.
    current_metrics = {
        "Total Cost": current_year_data["Annual_Cost"],
        "Total Budget": config.ROLLING_BUDGET[0]/5,  # Assuming first period's budget.
        "Avg Annual Capacity": current_year_data["Annual_Capacity"],
        "Avg Annual Capability": current_year_data["Annual_Capability"],
        "Avg annual KC135": current_year_data["KC135"],
        "Avg annual KC46": current_year_data["KC46"],
        "Avg annual KC46B": current_year_data["KC46B"],
        "Total Avg Annual Fleet": current_year_data["Total"]
    }
    
    # Convert the dictionary to a DataFrame with 2025 as the index.
    current_df = pd.DataFrame([current_metrics], index=["2025"])

    # Add the total fleet count (if not already present)
    df_sol["Total"] = df_sol["KC135"] + df_sol["KC46"] + df_sol["KC46B"]
    
    # Create a new column to mark the 5-year period.
    # We assume the simulation starts in 2025.
    start_year = df_sol["Year"].min()  # should be 2025
    df_sol["Period"] = ((df_sol["Year"] - start_year) // 5).astype(int)
    
    # Group by each 5-year period and compute the required statistics.
    period_groups = df_sol.groupby("Period")
    
    period_data = {}
    for period, group in period_groups:
        total_cost = group["Annual_Cost"].sum()
        # Here, "Total Budget" is taken from the config for that block.
        # (Assuming config.ROLLING_BUDGET is a list with one value per period.)
        total_budget = config.ROLLING_BUDGET[period] if period < len(config.ROLLING_BUDGET) else None
        avg_capacity = group["Annual_Capacity"].mean()
        avg_capability = group["Annual_Capability"].mean()
        avg_kc135 = group["KC135"].mean()
        avg_kc46 = group["KC46"].mean()
        avg_kc46b = group["KC46B"].mean()
        avg_total_fleet = group["Total"].mean()  # Total fleet average
    
        period_data[period] = {
            "Total Cost": total_cost,
            "Total Budget": total_budget,
            "Avg Annual Capacity": avg_capacity,
            "Avg Annual Capability": avg_capability,
            "Avg annual KC135": avg_kc135,
            "Avg annual KC46": avg_kc46,
            "Avg annual KC46B": avg_kc46b,
            "Total Avg Annual Fleet": avg_total_fleet
        }
    
    # Convert the period_data dictionary into a DataFrame.
    agg_df = pd.DataFrame(period_data).T
    
    # Create nice period labels, e.g. "2025-2029" for period 0.
    period_labels = {}
    for period in agg_df.index:
        period_start = start_year + period * 5
        period_end = period_start + 4
        period_labels[period] = f"{period_start}-{period_end}"
    agg_df.index = [period_labels[p] for p in agg_df.index]
    
    # Now, after creating your aggregated table (agg_df), combine it with the 2025 row.
    # This assumes agg_df's index contains period labels for subsequent periods.
    combined_df = pd.concat([current_df, agg_df])
    
    format_dict = {
        "Total Cost": "{:,.0f}",
        "Total Budget": "{:,.0f}",
        "Avg Annual Capacity": "{:,.2f}",
        "Avg Annual Capability": "{:,.2f}",
        "Avg annual KC135": "{:,.1f}",
        "Avg annual KC46": "{:,.1f}",
        "Avg annual KC46B": "{:,.1f}",
        "Total Avg Annual Fleet": "{:,.1f}"
    }

    # Using st.table with a styled DataFrame    
    st.subheader("Aggregated Metrics with 2025 Data")
    st.table(agg_df.style.format(format_dict))
