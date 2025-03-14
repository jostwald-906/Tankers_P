# ui.py
"""
User Interface module for the Tanker Fleet Optimization app.

This module sets up the Streamlit layout (tabs, sidebars, charts, and tables),
handles user inputs (GA parameters, model assumptions), executes the GA (with a progress bar),
and displays the simulation results along with additional decision-support sections.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import config
from ga import run_ga, toolbox
from simulate import simulate_fleet  # Note: our simulation functions are in simulate.py

def run_ui():
    """
    Launch the Streamlit UI.
    """
    st.set_page_config(layout="wide")
    st.title("Tanker Fleet Optimization with GA")
    
    # Create main tabs.
    tabs = st.tabs(["Model Info", "Run GA"])
    
    # -----------------------------
    # Model Info Tab
    # -----------------------------
    with tabs[0]:
        st.header("About the Model")
        st.markdown("""
**Inputs:**

- **Simulation Parameters:** Number of simulation years, block size for budgeting, minimum fleet size, discount rate, starting fleet counts, and rolling budgets.
- **Scenario Parameters:** Maintenance cost multipliers and capacity demands for peacetime, surge, and high‑threat.
- **Tanker Data:** Data for KC‑135, KC‑46, and KC46B (capacity, maintenance cost, procurement cost, max production per year, baseline capability metrics).
- **Upgrades:** Upgrade cost and improvements (delta values) for each tanker type.

**Approach Overview:**

Each point on the 3D plot represents one complete "plan" – a set of decisions (how many aircraft to buy, retire, upgrade each year). Plans are evaluated on:
- **Cost:** Total expenditure (maintenance, procurement, overhaul, and penalties) with future costs discounted.
- **Capacity:** Effective refueling capacity (adjusted for downtime).
- **Capability:** A composite operational performance score.

The GA evolves a population over many generations via selection, crossover, and mutation.
The final Pareto front shows the trade-offs between these objectives.

**Outputs:**

- A 3D Pareto front (with rotation animation).
- Stacked area charts showing fleet evolution.
- A detailed table of yearly decisions.
- A 5‑year aggregated table summarizing key metrics.
        """)
    
    # -----------------------------
    # Run GA Tab
    # -----------------------------
    with tabs[1]:
        # Sidebar: GA Parameters.
        st.sidebar.header("GA Parameters")
        pop_size_input = st.sidebar.number_input("Population Size", min_value=10, max_value=1000, 
                                                   value=config.DEFAULT_POP_SIZE,
                                                   help="Number of individuals in the GA population.")
        n_gen_input = st.sidebar.number_input("Number of Generations", min_value=5, max_value=100, 
                                              value=config.DEFAULT_NGEN,
                                              help="Number of generations for the GA.")
        cxpb_input = st.sidebar.slider("Crossover Probability", min_value=0.0, max_value=1.0, 
                                       value=config.DEFAULT_CXPB,
                                       help="Probability that two individuals will crossover.")
        mutpb_input = st.sidebar.slider("Mutation Probability", min_value=0.0, max_value=1.0, 
                                        value=config.DEFAULT_MUTPB,
                                        help="Probability that an individual will mutate.")
        
        # Model Assumptions Expander (collapsed by default).
        with st.expander("Model Assumptions (Click to Expand/Collapse)", expanded=False):
            new_min_fleet = st.number_input("Min Fleet Size", min_value=1, value=config.MIN_FLEET_SIZE,
                                            help="Minimum required fleet size.")
            new_discount = st.number_input("Discount Rate", min_value=0.0, max_value=1.0, step=0.01, 
                                           value=config.DISCOUNT_RATE,
                                           help="Rate used to discount future costs.")
            st.markdown("#### Scenario Parameters")
            new_peacetime_maint = st.number_input("Peacetime Maintenance Factor", 
                                                  value=config.SCENARIO_PARAMS["peacetime"]["maint_cost_factor"], 
                                                  step=0.1,
                                                  help="Multiplier for maintenance costs in peacetime.")
            new_peacetime_cap = st.number_input("Peacetime Capacity Demand", 
                                                value=config.SCENARIO_PARAMS["peacetime"]["capacity_demand"], 
                                                step=10,
                                                help="Required capacity in peacetime.")
            new_surge_maint = st.number_input("Surge Maintenance Factor", 
                                              value=config.SCENARIO_PARAMS["surge"]["maint_cost_factor"], 
                                              step=0.1,
                                              help="Maintenance cost multiplier during surge.")
            new_surge_cap = st.number_input("Surge Capacity Demand", 
                                            value=config.SCENARIO_PARAMS["surge"]["capacity_demand"], 
                                            step=10,
                                            help="Capacity demand during surge.")
            new_high_maint = st.number_input("High Threat Maintenance Factor", 
                                             value=config.SCENARIO_PARAMS["high_threat"]["maint_cost_factor"], 
                                             step=0.1,
                                             help="Maintenance cost multiplier under high threat.")
            new_high_cap = st.number_input("High Threat Capacity Demand", 
                                           value=config.SCENARIO_PARAMS["high_threat"]["capacity_demand"], 
                                           step=10,
                                           help="Capacity demand under high threat.")
            st.markdown("#### Tanker Data")
            st.markdown("##### KC135")
            new_kc135_capacity = st.number_input("KC135 Capacity", min_value=1, value=config.TANKER_DATA["KC135"]["capacity"], step=10,
                                                 help="Refueling capacity of a KC-135.")
            new_kc135_maint = st.number_input("KC135 Maintenance Cost", min_value=1, value=config.TANKER_DATA["KC135"]["maint_cost"], step=100000,
                                              help="Annual maintenance cost for a KC-135.")
            new_kc135_proc = st.number_input("KC135 Procurement Cost", min_value=1, value=config.TANKER_DATA["KC135"]["proc_cost"], step=1000000,
                                             help="Procurement cost for a KC-135.")
            new_kc135_max = st.number_input("KC135 Max Production", min_value=0, value=config.TANKER_DATA["KC135"]["max_production_per_year"], step=1,
                                            help="Maximum new KC-135s produced per year.")
            new_kc135_surv = st.number_input("KC135 Survivability", min_value=0.0, value=config.TANKER_DATA["KC135"]["cap_survivability"], step=0.05, format="%.2f",
                                             help="Baseline survivability for a KC-135.")
            new_kc135_conn = st.number_input("KC135 Connectivity", min_value=0.0, value=config.TANKER_DATA["KC135"]["cap_connectivity"], step=0.05, format="%.2f",
                                             help="Baseline connectivity for a KC-135.")
            new_kc135_mm = st.number_input("KC135 Multi-Mission", min_value=0.0, value=config.TANKER_DATA["KC135"]["cap_multi_mission"], step=0.05, format="%.2f",
                                           help="Baseline multi-mission capability for a KC-135.")
            
            st.markdown("##### KC46")
            new_kc46_capacity = st.number_input("KC46 Capacity", min_value=1, value=config.TANKER_DATA["KC46"]["capacity"], step=10,
                                                help="Refueling capacity of a KC-46.")
            new_kc46_maint = st.number_input("KC46 Maintenance Cost", min_value=1, value=config.TANKER_DATA["KC46"]["maint_cost"], step=100000,
                                             help="Annual maintenance cost for a KC-46.")
            new_kc46_proc = st.number_input("KC46 Procurement Cost", min_value=1, value=config.TANKER_DATA["KC46"]["proc_cost"], step=1000000,
                                            help="Procurement cost for a KC-46.")
            new_kc46_max = st.number_input("KC46 Max Production", min_value=0, value=config.TANKER_DATA["KC46"]["max_production_per_year"], step=1,
                                           help="Maximum new KC-46s produced per year.")
            new_kc46_surv = st.number_input("KC46 Survivability", min_value=0.0, value=config.TANKER_DATA["KC46"]["cap_survivability"], step=0.05, format="%.2f",
                                            help="Baseline survivability for a KC-46.")
            new_kc46_conn = st.number_input("KC46 Connectivity", min_value=0.0, value=config.TANKER_DATA["KC46"]["cap_connectivity"], step=0.05, format="%.2f",
                                            help="Baseline connectivity for a KC-46.")
            new_kc46_mm = st.number_input("KC46 Multi-Mission", min_value=0.0, value=config.TANKER_DATA["KC46"]["cap_multi_mission"], step=0.05, format="%.2f",
                                          help="Baseline multi-mission capability for a KC-46.")
            
            st.markdown("##### KC46B")
            new_KC46B_capacity = st.number_input("KC46B Capacity", min_value=1, value=config.TANKER_DATA["KC46B"]["capacity"], step=10,
                                                help="Refueling capacity of an KC46B aircraft.")
            new_KC46B_maint = st.number_input("KC46B Maintenance Cost", min_value=1, value=config.TANKER_DATA["KC46B"]["maint_cost"], step=100000,
                                             help="Annual maintenance cost for KC46B.")
            new_KC46B_proc = st.number_input("KC46B Procurement Cost", min_value=1, value=config.TANKER_DATA["KC46B"]["proc_cost"], step=1000000,
                                            help="Procurement cost for an KC46B aircraft.")
            new_KC46B_max = st.number_input("KC46B Max Production", min_value=0, value=config.TANKER_DATA["KC46B"]["max_production_per_year"], step=1,
                                           help="Maximum new KC46B produced per year.")
            new_KC46B_surv = st.number_input("KC46B Survivability", min_value=0.0, value=config.TANKER_DATA["KC46B"]["cap_survivability"], step=0.05, format="%.2f",
                                            help="Baseline survivability for KC46B.")
            new_KC46B_conn = st.number_input("KC46B Connectivity", min_value=0.0, value=config.TANKER_DATA["KC46B"]["cap_connectivity"], step=0.05, format="%.2f",
                                            help="Baseline connectivity for KC46B.")
            new_KC46B_mm = st.number_input("KC46B Multi-Mission", min_value=0.0, value=config.TANKER_DATA["KC46B"]["cap_multi_mission"], step=0.05, format="%.2f",
                                          help="Baseline multi-mission capability for KC46B.")
            
            st.markdown("#### Upgrades")
            st.markdown("##### KC135")
            new_kc135_upg_cost = st.number_input("KC135 Upgrade Cost", min_value=1, value=config.UPGRADES["KC135"]["cost"], step=100000,
                                                 help="Cost to upgrade a KC-135.")
            new_kc135_delta_surv = st.number_input("KC135 Delta Survivability", min_value=0.0, 
                                                   value=config.UPGRADES["KC135"]["delta_survivability"], 
                                                   step=0.05, format="%.2f",
                                                   help="Increase in survivability after upgrade for a KC-135.")
            new_kc135_delta_conn = st.number_input("KC135 Delta Connectivity", min_value=0.0, 
                                                   value=config.UPGRADES["KC135"]["delta_connectivity"], 
                                                   step=0.05, format="%.2f",
                                                   help="Increase in connectivity after upgrade for a KC-135.")
            new_kc135_delta_mm = st.number_input("KC135 Delta Multi-Mission", min_value=0.0, 
                                                 value=config.UPGRADES["KC135"]["delta_multi_mission"], 
                                                 step=0.05, format="%.2f",
                                                 help="Increase in multi-mission capability for a KC-135 after upgrade.")
            
            st.markdown("##### KC46")
            new_kc46_upg_cost = st.number_input("KC46 Upgrade Cost", min_value=1, value=config.UPGRADES["KC46"]["cost"], step=100000,
                                                help="Cost to upgrade a KC-46.")
            new_kc46_delta_surv = st.number_input("KC46 Delta Survivability", min_value=0.0, 
                                                  value=config.UPGRADES["KC46"]["delta_survivability"], 
                                                  step=0.05, format="%.2f",
                                                  help="Increase in survivability for a KC-46 after upgrade.")
            new_kc46_delta_conn = st.number_input("KC46 Delta Connectivity", min_value=0.0, 
                                                  value=config.UPGRADES["KC46"]["delta_connectivity"], 
                                                  step=0.05, format="%.2f",
                                                  help="Increase in connectivity for a KC-46 after upgrade.")
            new_kc46_delta_mm = st.number_input("KC46 Delta Multi-Mission", min_value=0.0, 
                                                value=config.UPGRADES["KC46"]["delta_multi_mission"], 
                                                step=0.05, format="%.2f",
                                                help="Increase in multi-mission capability for a KC-46 after upgrade.")
            
            st.markdown("##### KC46B")
            new_KC46B_upg_cost = st.number_input("KC46B Upgrade Cost", min_value=1, value=config.UPGRADES["KC46B"]["cost"], step=100000,
                                                help="Cost to upgrade an KC46B aircraft.")
            new_KC46B_delta_surv = st.number_input("KC46B Delta Survivability", min_value=0.0, 
                                                  value=config.UPGRADES["KC46B"]["delta_survivability"], 
                                                  step=0.05, format="%.2f",
                                                  help="Increase in survivability for KC46B after upgrade.")
            new_KC46B_delta_conn = st.number_input("KC46B Delta Connectivity", min_value=0.0, 
                                                  value=config.UPGRADES["KC46B"]["delta_connectivity"], 
                                                  step=0.05, format="%.2f",
                                                  help="Increase in connectivity for KC46B after upgrade.")
            new_KC46B_delta_mm = st.number_input("KC46B Delta Multi-Mission", min_value=0.0, 
                                                value=config.UPGRADES["KC46B"]["delta_multi_mission"], 
                                                step=0.05, format="%.2f",
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
                
                for key in ['pareto', 'costs', 'caps', 'cabs', 'timeseries_data']:
                    if key in st.session_state:
                        del st.session_state[key]
                
                st.success("Model assumptions updated.")
        
        # --- GA Execution ---
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
            
            progress_bar = st.progress(0)
            pop, pareto = run_ga(progress_bar)
            progress_bar.empty()
            
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
                df["Procurement_Cost"] = (
                    df["KC135_bought"] * config.TANKER_DATA["KC135"]["proc_cost"] +
                    df["KC46_bought"] * config.TANKER_DATA["KC46"]["proc_cost"] +
                    df["KC46B_bought"] * config.TANKER_DATA["KC46B"]["proc_cost"]
                )
                df["Operational_Cost"] = df["Annual_Cost"] - df["Procurement_Cost"]
                timeseries_data[i] = df
            st.session_state['timeseries_data'] = timeseries_data
            
            pool.close()
            pool.join()
            st.success("GA Completed.")
        
        # --- Results Display ---
        if 'pareto' in st.session_state:
            # 3D Pareto Front Plot with Rotation Animation.
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
            
            num_frames = 120
            angles = np.linspace(0, 360, num_frames)
            frames = []
            for angle in angles:
                camera = dict(
                    eye=dict(
                        x=1.5 * np.cos(np.radians(angle)),
                        y=1.5 * np.sin(np.radians(angle)),
                        z=2
                    )
                )
                frames.append(go.Frame(layout=dict(scene_camera=camera)))
            fig3d.frames = frames
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
            
            # Allow user to select a solution.
            sol_index = st.selectbox("Select a Pareto Solution for Detailed Analysis", list(range(len(st.session_state['costs']))))
            df_sol = st.session_state['timeseries_data'][sol_index]
            
            # Fleet Composition Area Chart.
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
            
            # Yearly Decision Details Table.
            required_cols = ["Year", "KC135_bought", "KC135_retired", "KC135_upgraded",
                             "KC46_bought", "KC46_retired", "KC46_upgraded",
                             "KC46B_bought", "KC46B_retired", "KC46B_upgraded", "Total"]
            for col in required_cols:
                if col not in df_sol.columns:
                    df_sol[col] = 0
            st.subheader("Yearly Decision Details")
            st.table(df_sol[required_cols])
            
            # 5-Year Aggregated Table.
            with st.expander("5-Year Aggregated Metrics", expanded=True):
                st.markdown("### Aggregated Metrics Over 5-Year Periods")
                df_agg = df_sol.copy()
                df_agg["Total"] = df_agg["KC135"] + df_agg["KC46"] + df_agg["KC46B"]
                start_year = df_agg["Year"].min()  # should be 2025
                df_agg["Period"] = ((df_agg["Year"] - start_year) // 5).astype(int)
                period_groups = df_agg.groupby("Period")
                agg_data = {}
                for period, group in period_groups:
                    total_cost = group["Annual_Cost"].sum()
                    total_budget = config.ROLLING_BUDGET[period] if period < len(config.ROLLING_BUDGET) else None
                    avg_capacity = group["Annual_Capacity"].mean()
                    avg_capability = group["Annual_Capability"].mean()
                    avg_kc135 = group["KC135"].mean()
                    avg_kc46 = group["KC46"].mean()
                    avg_kc46b = group["KC46B"].mean()
                    avg_total_fleet = group["Total"].mean()
                    agg_data[period] = {
                        "Total Cost": total_cost,
                        "Total Budget": total_budget,
                        "Avg Annual Capacity": avg_capacity,
                        "Avg Annual Capability": avg_capability,
                        "Avg annual KC135": avg_kc135,
                        "Avg annual KC46": avg_kc46,
                        "Avg annual KC46B": avg_kc46b,
                        "Total Avg Annual Fleet": avg_total_fleet
                    }
                agg_df = pd.DataFrame(agg_data).T
                # Create period labels.
                period_labels = {}
                for period in agg_df.index:
                    period_start = start_year + period * 5
                    period_end = period_start + 4
                    period_labels[period] = f"{period_start}-{period_end}"
                agg_df.index = [period_labels[p] for p in agg_df.index]
                
                # Reinsert current year values as the first row.
                current_year_data = df_sol[df_sol["Year"] == start_year].iloc[0]
                current_metrics = {
                    "Total Cost": current_year_data["Annual_Cost"],
                    "Total Budget": config.ROLLING_BUDGET[0] / 5,  # Assuming budget is evenly allocated per year.
                    "Avg Annual Capacity": current_year_data["Annual_Capacity"],
                    "Avg Annual Capability": current_year_data["Annual_Capability"],
                    "Avg annual KC135": current_year_data["KC135"],
                    "Avg annual KC46": current_year_data["KC46"],
                    "Avg annual KC46B": current_year_data["KC46B"],
                    "Total Avg Annual Fleet": current_year_data["Total"]
                }
                current_df = pd.DataFrame([current_metrics], index=[str(start_year)])
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
                st.table(combined_df.style.format(format_dict))
            
            # Dashboard Summary: Tiles for selected solution and Pareto averages.
            with st.expander("Dashboard Summary (Key Performance Indicators)", expanded=True):
                st.markdown("### Dashboard Summary")
                selected_solution = st.session_state['pareto'][sol_index]
                # Selected solution KPIs.
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Cost", f"${selected_solution.fitness.values[0]:,.0f}")
                col2.metric("Total Capacity", f"{selected_solution.fitness.values[1]:,.2f}")
                col3.metric("Total Capability", f"{selected_solution.fitness.values[2]:,.2f}")
                st.markdown("#### Pareto Front Averages")
                avg_cost = np.mean([ind.fitness.values[0] for ind in st.session_state['pareto']])
                avg_cap = np.mean([ind.fitness.values[1] for ind in st.session_state['pareto']])
                avg_capab = np.mean([ind.fitness.values[2] for ind in st.session_state['pareto']])
                col4, col5, col6 = st.columns(3)
                col4.metric("Avg Total Cost", f"${avg_cost:,.0f}")
                col5.metric("Avg Total Capacity", f"{avg_cap:,.2f}")
                col6.metric("Avg Total Capability", f"{avg_capab:,.2f}")
            
            # Sensitivity Analysis Section.
            with st.expander("Sensitivity Analysis", expanded=True):
                st.markdown("### Sensitivity Analysis")
                st.write("Select a parameter to vary and an output metric to view its effect.")
                param_option = st.selectbox("Parameter to Vary", 
                    options=["Discount Rate", "Procurement Cost (KC135)", "Procurement Cost (KC46)", "Procurement Cost (KC46B)",
                             "Upgrade Cost (KC135)", "Upgrade Cost (KC46)", "Upgrade Cost (KC46B)"])
                output_option = st.selectbox("Output Metric", options=["Total Cost", "Total Capacity", "Total Capability"])
                
                selected_solution = st.session_state['pareto'][sol_index]
                test_values = []
                output_values = []
                original_value = None
                if param_option == "Discount Rate":
                    original_value = config.DISCOUNT_RATE
                    test_values = np.linspace(original_value - 0.05, original_value + 0.05, 10)
                elif param_option.startswith("Procurement Cost"):
                    ac_type = param_option.split("(")[1].replace(")", "")
                    original_value = config.TANKER_DATA[ac_type]["proc_cost"]
                    test_values = np.linspace(original_value * 0.8, original_value * 1.2, 10)
                elif param_option.startswith("Upgrade Cost"):
                    ac_type = param_option.split("(")[1].replace(")", "")
                    original_value = config.UPGRADES[ac_type]["cost"]
                    test_values = np.linspace(original_value * 0.8, original_value * 1.2, 10)
                
                for val in test_values:
                    if param_option == "Discount Rate":
                        config.DISCOUNT_RATE = val
                    elif param_option.startswith("Procurement Cost"):
                        config.TANKER_DATA[ac_type]["proc_cost"] = val
                    elif param_option.startswith("Upgrade Cost"):
                        config.UPGRADES[ac_type]["cost"] = val
                    result = simulate_fleet(selected_solution)
                    if output_option == "Total Cost":
                        output_values.append(result[0])
                    elif output_option == "Total Capacity":
                        output_values.append(result[1])
                    elif output_option == "Total Capability":
                        output_values.append(result[2])
                if param_option == "Discount Rate":
                    config.DISCOUNT_RATE = original_value
                elif param_option.startswith("Procurement Cost"):
                    config.TANKER_DATA[ac_type]["proc_cost"] = original_value
                elif param_option.startswith("Upgrade Cost"):
                    config.UPGRADES[ac_type]["cost"] = original_value
                
                sens_fig = go.Figure()
                sens_fig.add_trace(go.Scatter(x=test_values, y=output_values, mode='lines+markers'))
                sens_fig.update_layout(
                    title=f"Sensitivity of {output_option} to {param_option}",
                    xaxis_title=param_option,
                    yaxis_title=output_option
                )
                st.plotly_chart(sens_fig, use_container_width=True)
    
if __name__ == "__main__":
    run_ui()
