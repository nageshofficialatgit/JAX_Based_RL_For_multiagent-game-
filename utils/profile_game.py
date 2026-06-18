import cProfile
import pstats
from training.arena import simulate
import braniac_v2

if __name__ == '__main__':
    agents = [braniac_v2.agent] * 4
    profiler = cProfile.Profile()
    profiler.enable()
    simulate(agents, seed=42, max_steps=100)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(30)
