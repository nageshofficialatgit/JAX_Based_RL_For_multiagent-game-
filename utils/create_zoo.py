import json
import os

base = {
    "my_production_base": 15.0, "my_production_decay": -0.05, "my_planets": 5.0,
    "enemy_ships_total": -0.2, "enemy_prod_total": -2.0, "enemy_leader_ships": -0.5,
    "we_are_leader_penalty": -50.0, "my_spread_penalty": -0.5, "enemy_spread_bonus": 1.0,
    "weight_t30": 0.5, "weight_t60": 0.3, "weight_t100": 0.2,
    "dist_penalty": -1.0, "target_enemy_prod_bonus": 10.0, "target_neutral_cost_penalty": -0.5,
    "friendly_defense_bonus": 10.0, "retreat_threshold": 1.5
}

zoo = {
    "zoo_turtle": {**base, "retreat_threshold": 3.0, "friendly_defense_bonus": 50.0, "my_planets": 10.0, "target_enemy_prod_bonus": 0.0},
    "zoo_swarm": {**base, "retreat_threshold": 0.8, "dist_penalty": -0.1, "my_production_base": 5.0, "target_neutral_cost_penalty": 0.0},
    "zoo_boomer": {**base, "my_production_base": 50.0, "my_production_decay": 0.0, "my_planets": 0.0},
    "zoo_sniper": {**base, "target_enemy_prod_bonus": 20.0, "dist_penalty": -0.1, "target_neutral_cost_penalty": -2.0, "retreat_threshold": 0.8}
}

os.makedirs("training/archive", exist_ok=True)
for name, params in zoo.items():
    with open(f"training/archive/{name}.json", "w") as f:
        json.dump(params, f, indent=2)
print("Zoo created!")
