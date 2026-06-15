import json
import os

dummy_replay = {
    "rewards": [100.0, -100.0],
    "info": {
        "seed": 42,
        "TeamNames": ["BotA", "BotB"]
    },
    "steps": [
        [
            {
                "observation": {
                    "angular_velocity": 0.05,
                    "comet_planet_ids": [2],
                    "planets": [
                        [0, 0, 10.0, 10.0, 5.0, 50, 10],
                        [1, 1, 90.0, 90.0, 5.0, 50, 10],
                        [2, -1, 50.0, 50.0, 2.0, 0, 0]
                    ],
                    "fleets": []
                }
            },
            {"observation": {}}
        ],
        [
            {
                "observation": {
                    "angular_velocity": 0.05,
                    "planets": [
                        [0, 0, 10.0, 10.0, 5.0, 60, 10],
                        [1, 1, 90.0, 90.0, 5.0, 60, 10]
                    ],
                    "fleets": [
                        [0, 0, 1, 20.0, 20.0, 1.0, 25, 10]
                    ]
                },
                "action": [[0, 45.0, 25]]
            },
            {"action": []}
        ]
    ]
}

os.makedirs("raw_replays", exist_ok=True)
with open("raw_replays/episode-12345.json", "w") as f:
    json.dump(dummy_replay, f)

print("Generated dummy raw_replays/episode-12345.json")
