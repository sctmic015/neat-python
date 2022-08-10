from simulator import Simulator
from controllers.kinematic import Controller, tripod_gait, stationary, wave_gait, quadruped_gait

controller = Controller(quadruped_gait, body_height=0.15, velocity=0.46, crab_angle=-1.57)
simulator = Simulator(controller, follow=True, visualiser=True, collision_fatal=False, failed_legs=[0])

# run indefinitely
while True:
	simulator.step()
