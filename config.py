# config.py

# Simulation parameters
YEARS = 30
BLOCK_SIZE = 5            # for rolling 5-year budgets
MIN_FLEET_SIZE = 466
DISCOUNT_RATE = 0.02       # discount future costs

# Default GA parameters
DEFAULT_POP_SIZE = 100
DEFAULT_NGEN = 30
DEFAULT_CXPB = 0.8
DEFAULT_MUTPB = 0.2

# Scenarios by year
SCENARIOS = []
for y in range(YEARS):
    if y < 5:
        SCENARIOS.append("peacetime")
    elif y < 10:
        SCENARIOS.append("surge")
    elif y < 20:
        SCENARIOS.append("high_threat")
    else:
        SCENARIOS.append("peacetime")

SCENARIO_PARAMS = {
    "peacetime": {"maint_cost_factor": 1.0, "capacity_demand": 1000},
    "surge": {"maint_cost_factor": 1.3, "capacity_demand": 1500},
    "high_threat": {"maint_cost_factor": 1.6, "capacity_demand": 2000}
}

ROLLING_BUDGET = [5e9, 4e9, 5e9, 6e9, 5e9, 4e9]  # six blocks for 30 years

BASE_OVERHAUL_PROB = {"KC135": 0.1, "KC46": 0.05, "NGAS": 0.03}
SUBCOMPONENT_WEIGHTS = {"survivability": 0.4, "connectivity": 0.3, "multi_mission": 0.3}

# Tanker specifications
TANKER_DATA = {
    "KC135": {
        "capacity": 200,
        "cap_survivability": 0.4,
        "cap_connectivity": 0.3,
        "cap_multi_mission": 0.2,
        "maint_cost": 4_000_000,
        "proc_cost": 50_000_000,
        "max_production_per_year": 0
    },
    "KC46": {
        "capacity": 220,
        "cap_survivability": 0.7,
        "cap_connectivity": 0.5,
        "cap_multi_mission": 0.2,
        "maint_cost": 3_000_000,
        "proc_cost": 200_000_000,
        "max_production_per_year": 20
    },
    "NGAS": {
        "capacity": 250,
        "cap_survivability": 0.9,
        "cap_connectivity": 0.6,
        "cap_multi_mission": 0.3,
        "maint_cost": 2_500_000,
        "proc_cost": 300_000_000,
        "max_production_per_year": 10
    }
}

# Upgrade costs and improvements
UPGRADES = {
    "KC135": {"cost": 10_000_000, "delta_survivability": 0.1, "delta_connectivity": 0.2, "delta_multi_mission": 0.1},
    "KC46": {"cost": 20_000_000, "delta_survivability": 0.1, "delta_connectivity": 0.2, "delta_multi_mission": 0.0},
    "NGAS": {"cost": 30_000_000, "delta_survivability": 0.05, "delta_connectivity": 0.1, "delta_multi_mission": 0.2}
}

# Starting fleet counts
START_KC135_COUNT = 400
START_KC46_COUNT = 66
START_NGAS_COUNT = 0
USAGE_MAX = 3
