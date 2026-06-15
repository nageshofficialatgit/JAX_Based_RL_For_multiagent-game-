import jax
import jax.numpy as jnp
from orbit_env import EnvState, get_fleet_speed, step_physics

def test_fleet_speed():
    ships = jnp.array([1.0, 10.0, 1000.0])
    speeds = get_fleet_speed(ships)
    print(f"Ships: {ships} -> Speeds: {speeds}")
    assert jnp.allclose(speeds[0], 1.0)
    assert jnp.allclose(speeds[2], 6.0)
    print("test_fleet_speed passed!")

def test_basic_step():
    state = EnvState(
        planet_x=jnp.array([10.0]),
        planet_y=jnp.array([10.0]),
        planet_initial_x=jnp.array([10.0]),
        planet_initial_y=jnp.array([10.0]),
        planet_is_orbiting=jnp.array([1]), # Test rotating
        angular_velocity=jnp.array(0.05),
        planet_radius=jnp.array([5.0]),
        planet_production=jnp.array([1.5]),
        planet_owner=jnp.array([1]),
        planet_ships=jnp.array([100.0]),
        
        fleet_active=jnp.array([1]),
        fleet_owner=jnp.array([2]),
        fleet_ships=jnp.array([50.0]),
        fleet_x=jnp.array([0.0]),
        fleet_y=jnp.array([0.0]),
        fleet_dx=jnp.array([1.0]),
        fleet_dy=jnp.array([0.0]),
        fleet_src_planet=jnp.array([0]),
        
        tick=jnp.array(0)
    )
    
    next_state = step_physics(state)
    print(f"Planet Ships: {state.planet_ships} -> {next_state.planet_ships}")
    print(f"Fleet X: {state.fleet_x} -> {next_state.fleet_x}")
    
    assert jnp.allclose(next_state.planet_ships[0], 101.5)
    assert jnp.allclose(next_state.fleet_x[0], 1.0)
    print("test_basic_step passed!")

def test_combat_resolution():
    state = EnvState(
        # 1 planet at (10, 10), radius 5, owner 1 (P1), 10 ships
        planet_x=jnp.array([10.0]),
        planet_y=jnp.array([10.0]),
        planet_initial_x=jnp.array([10.0]),
        planet_initial_y=jnp.array([10.0]),
        planet_is_orbiting=jnp.array([0]), # Static for combat test
        angular_velocity=jnp.array(0.0),
        planet_radius=jnp.array([5.0]),
        planet_production=jnp.array([0.0]),
        planet_owner=jnp.array([1]),
        planet_ships=jnp.array([10.0]),
        
        # 2 fleets arriving at (10, 10)
        # Fleet 0: P2, 30 ships. Fleet 1: P3, 15 ships.
        fleet_active=jnp.array([1, 1]),
        fleet_owner=jnp.array([2, 3]),
        fleet_ships=jnp.array([30.0, 15.0]),
        fleet_x=jnp.array([10.0, 10.0]),
        fleet_y=jnp.array([10.0, 10.0]),
        fleet_dx=jnp.array([0.0, 0.0]),
        fleet_dy=jnp.array([0.0, 0.0]),
        fleet_src_planet=jnp.array([0, 1]),
        
        tick=jnp.array(0)
    )
    
    next_state = step_physics(state)
    print("Combat Test:")
    print(f"Initial: P1 has 10 ships. P2 attacks with 30. P3 attacks with 15.")
    print(f"Top 1 (P2): 30. Top 2 (P3): 15.")
    print(f"Expected Survivor: P2 with 5 ships (15 orbital survivor - 10 garrison).")
    print(f"Final Owner: {next_state.planet_owner[0]}, Final Ships: {next_state.planet_ships[0]}")
    print(f"Fleets Active: {next_state.fleet_active}")
    
    assert next_state.planet_owner[0] == 2
    assert jnp.allclose(next_state.planet_ships[0], 5.0)
    assert jnp.all(next_state.fleet_active == 0) # Both fleets died
    print("test_combat_resolution passed!")

if __name__ == "__main__":
    test_fleet_speed()
    test_basic_step()
    test_combat_resolution()
