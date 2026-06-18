from kaggle_environments import make
env = make("orbit_wars", configuration={"episodeSteps": 10}, debug=True)
trainer = env.train([None, "random"])
obs = trainer.reset()
print("raw player:", obs.player)
print("owner of my planet:", [p[1] for p in obs.planets if p[1] != -1])
