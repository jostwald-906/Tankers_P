# simulation.py
"""
Simulation module for the Tanker Fleet Optimization model.

This module defines functions that simulate a multi-year fleet plan.
Each individual solution (a list of yearly decisions) is simulated to produce:
  - Total discounted cost (with penalties if constraints are violated)
  - Total effective capacity over the planning horizon
  - Total capability score over the planning horizon
  - Detailed yearly metrics (including effective actions)
"""

import random, math, copy
import numpy as np
import pandas as pd
import config

def check_overhaul(aircraft, base_prob):
    """
    Determine if an aircraft should undergo an overhaul.
    
    The probability increases with the aircraft's age.
    """
    prob = base_prob + 0.01 * (aircraft["age"] // 10)
    return random.random() < prob

def simulate_fleet(individual):
    """
    Simulate the fleet plan for an individual solution over the planning horizon.
    
    Parameters:
      individual: A list of yearly decisions. Each decision is a list:
          [buy_kc135, upgrade_kc135, retire_kc135,
           buy_kc46, upgrade_kc46, retire_kc46,
           buy_KC46B, upgrade_KC46B, retire_KC46B,
           usage_profile]
    
    Returns a tuple:
      (total_cost, total_capacity, total_capability, year_data)
      
    where year_data is a list of dictionaries with per-year fleet metrics.
    """
    # Build initial fleets using starting counts from config.
    kc135_fleet = [{
        "age": 50,
        "downtime": 0.0,
        "survivability": config.TANKER_DATA["KC135"]["cap_survivability"],
        "connectivity": config.TANKER_DATA["KC135"]["cap_connectivity"],
        "multi_mission": config.TANKER_DATA["KC135"]["cap_multi_mission"]
    } for _ in range(config.START_KC135_COUNT)]
    
    kc46_fleet = [{
        "age": 5,
        "downtime": 0.0,
        "survivability": config.TANKER_DATA["KC46"]["cap_survivability"],
        "connectivity": config.TANKER_DATA["KC46"]["cap_connectivity"],
        "multi_mission": config.TANKER_DATA["KC46"]["cap_multi_mission"]
    } for _ in range(config.START_KC46_COUNT)]
    
    KC46B_fleet = [{
        "age": 0,
        "downtime": 0.0,
        "survivability": config.TANKER_DATA["KC46B"]["cap_survivability"],
        "connectivity": config.TANKER_DATA["KC46B"]["cap_connectivity"],
        "multi_mission": config.TANKER_DATA["KC46B"]["cap_multi_mission"]
    } for _ in range(config.START_KC46B_COUNT)]
    
    # Recompute SCENARIOS based on config.YEARS.
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
    
    # Adjust ROLLING_BUDGET to match the number of 5-year blocks.
    num_blocks = math.ceil(config.YEARS / config.BLOCK_SIZE)
    if len(config.ROLLING_BUDGET) != num_blocks:
        config.ROLLING_BUDGET = (config.ROLLING_BUDGET * num_blocks)[:num_blocks]
    
    used_budget_blocks = [0.0] * len(config.ROLLING_BUDGET)
    total_cost = 0.0
    total_capacity = 0.0
    total_capability = 0.0
    year_data = []
    
    # Process each year's decision.
    for year, decision in enumerate(individual):
        block_idx = year // config.BLOCK_SIZE
        scenario = config.SCENARIOS[year]
        (buy_kc135, upgrade_kc135, retire_kc135,
         buy_kc46, upgrade_kc46, retire_kc46,
         buy_KC46B, upgrade_KC46B, retire_KC46B,
         usage_profile) = decision
        
        # Enforce production limits.
        buy_kc135 = min(buy_kc135, config.TANKER_DATA["KC135"]["max_production_per_year"])
        buy_kc46  = min(buy_kc46, config.TANKER_DATA["KC46"]["max_production_per_year"])
        buy_KC46B = min(buy_KC46B, config.TANKER_DATA["KC46B"]["max_production_per_year"])
        if year < 10:
            buy_KC46B = 0  # No KC46B purchase before year 10.
        
        # Store initial fleet counts before retirements.
        initial_kc135 = len(kc135_fleet)
        initial_kc46  = len(kc46_fleet)
        initial_KC46B = len(KC46B_fleet)
        
        # Compute effective retirements.
        effective_retire_kc135 = min(retire_kc135, initial_kc135)
        effective_retire_kc46  = min(retire_kc46, initial_kc46)
        effective_retire_KC46B = min(retire_KC46B, initial_KC46B)
        
        # Process retirements.
        for _ in range(effective_retire_kc135):
            kc135_fleet.pop(0)
        for _ in range(effective_retire_kc46):
            kc46_fleet.pop(0)
        for _ in range(effective_retire_KC46B):
            KC46B_fleet.pop(0)
        
        # Process buy actions.
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
        
        # Compute effective upgrades (only if an aircraft is available).
        effective_upgrade_kc135 = 1 if (upgrade_kc135 > 0 and len(kc135_fleet) > 0) else 0
        effective_upgrade_kc46  = 1 if (upgrade_kc46 > 0 and len(kc46_fleet) > 0) else 0
        effective_upgrade_KC46B = 1 if (upgrade_KC46B > 0 and len(KC46B_fleet) > 0) else 0
        
        # Apply upgrades and update budget.
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
        
        # Calculate annual metrics.
        annual_cost = 0.0
        annual_capacity = 0.0
        annual_capability = 0.0
        maint_factor = config.SCENARIO_PARAMS[scenario]["maint_cost_factor"]
        demand = config.SCENARIO_PARAMS[scenario]["capacity_demand"]
        
        def process_fleet(fleet, type_key):
            nonlocal annual_cost, annual_capacity, annual_capability
            base_maint = config.TANKER_DATA[type_key]["maint_cost"]
            base_cap = config.TANKER_DATA[type_key]["capacity"]
            for ac in fleet:
                # Increment aircraft age.
                ac["age"] += 1
                # Check for overhaul.
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
                ac["downtime"] = 0.0  # Reset downtime after calculation.
                annual_capacity += base_cap * (1.0 - downtime_frac)
                # Compute composite capability score.
                c_score = (ac["survivability"] * config.SUBCOMPONENT_WEIGHTS["survivability"] +
                           ac["connectivity"] * config.SUBCOMPONENT_WEIGHTS["connectivity"] +
                           ac["multi_mission"] * config.SUBCOMPONENT_WEIGHTS["multi_mission"])
                annual_capability += c_score * (1.0 - downtime_frac)
        
        process_fleet(kc135_fleet, "KC135")
        process_fleet(kc46_fleet, "KC46")
        process_fleet(KC46B_fleet, "KC46B")
        
        # Add procurement costs.
        proc_cost_kc135 = config.TANKER_DATA["KC135"]["proc_cost"] * buy_kc135
        proc_cost_kc46  = config.TANKER_DATA["KC46"]["proc_cost"] * buy_kc46
        proc_cost_KC46B = config.TANKER_DATA["KC46B"]["proc_cost"] * buy_KC46B
        year_proc_cost = proc_cost_kc135 + proc_cost_kc46 + proc_cost_KC46B
        used_budget_blocks[block_idx] += year_proc_cost
        annual_cost += year_proc_cost
        
        # Discount the annual cost.
        annual_cost = annual_cost / ((1 + config.DISCOUNT_RATE) ** year)
        total_cost += annual_cost
        
        # Apply penalties if constraints are violated.
        fleet_size = len(kc135_fleet) + len(kc46_fleet) + len(KC46B_fleet)
        if fleet_size < config.MIN_FLEET_SIZE:
            total_cost += 1e15
        if annual_capacity < demand:
            total_cost += 1e9
        
        total_capacity += annual_capacity
        total_capability += annual_capability
        
        # Record this year's data.
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
    """
    Evaluate an individual solution by simulating its fleet plan.
    The individual's fitness is set to (total_cost, total_capacity, total_capability)
    and its simulation data is attached.
    """
    cost, capacity, capability, year_data = simulate_fleet(ind)
    ind.fitness.values = (cost, capacity, capability)
    ind.year_data = copy.deepcopy(year_data)
    return ind
