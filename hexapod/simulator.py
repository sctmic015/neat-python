import pybullet_utils.bullet_client as bc
import pybullet as p
import pybullet_data
import numpy as np
import time
import os


class Simulator:
	"""This class is a wrapper for simulating the RARL Hexapod with the PyBullet physics engine

	Note
	----
	This class is configured to automatically start a PyBullet instance on an available CPU core.
	Each initialised class will use a separate CPU aiding parallelisation

    Attributes
    ----------    	
    urdf : str
    	Filename of URDF model of hexapod relative to simulator. 
    controller : :obj:'Controller'
    	Positional controller object responsible for providing joint angles and velocities at given time
    visualiser_enabled : bool
    	Whether to display a GUI
    collision_fatal : bool
    	Whether collisions between links raise an exception
    locked_joints : :obj:`list` of :obj:`int`
    	A list of joint numbers which should be fixed in a retracted position
    failed_joints : :obj:`list` of :obj:`int`
    	A list of joint numbers which should simulated as unpowered.

    """
	def __init__(self, controller, urdf='/urdf/hexapod_simplified.urdf', visualiser=False, follow=True, collision_fatal=True, failed_legs=[], camera_position=[0, 0, 0], camera_distance=0.7, camera_yaw=20, camera_pitch=-30):
		self.t = 0 #: float: Current time of the simulator
		self.dt = 1.0/ 240  #: float: Timestep of simulator. Default is 1/240s for PyBullet.
		self.previous_joint_angles = []
		self.count = 0
		self.n_step = 0
		self.gravity = -9.81 #: float: Magnitude of gravity vector in the positive z direction
		self.foot_friction = 0.7
		self.controller = controller
		self.visualiser_enabled = visualiser
		self.follow = follow
		self.collision_fatal = collision_fatal
		self.failed_joints = []
		# set up joints to be locked to simulate failure
		self.locked_joints = []
		for failed_leg in failed_legs:
			self.locked_joints += [failed_leg*3-3, failed_leg*3-2, failed_leg*3-1]

		self.camera_position = [0.7, 0, 0] #: list of float: GUI camera focus position in cartesian coordinates (self.controller.body_height)
		self.camera_distance = 0.8 #: float: GUI camera distance from camera position
		self.camera_yaw = 0 #: float: GUI camera yaw in degrees (-20)
		self.camera_pitch = -45 #: float: GUI camera pitch in degrees (-30)

		self.camera_position = camera_position
		self.camera_distance = camera_distance
		self.camera_yaw = camera_yaw
		self.camera_pitch = camera_pitch
		
		# connect a client to a pybullet physics server
		if self.visualiser_enabled:
			self.client = bc.BulletClient(connection_mode=p.GUI)
			self.client.resetDebugVisualizerCamera(cameraDistance=self.camera_distance, cameraYaw=self.camera_yaw, cameraPitch=self.camera_pitch, cameraTargetPosition=self.camera_position)
			self.client.configureDebugVisualizer(p.COV_ENABLE_GUI, False) # somestimes useful to turn on
			self.client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
			self.client.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, False, shadowMapResolution=8192) # for nice rendering
			self.client.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, False) # somestimes useful to turn on
			self.client.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, False)
			self.client.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, False)
			self.client.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, False)
		else:
			self.client = bc.BulletClient(connection_mode=p.DIRECT) # Prints here

		self.client.setAdditionalSearchPath(pybullet_data.getDataPath())
		self.client.setGravity(0, 0, self.gravity)
		self.client.setRealTimeSimulation(False) # simulation needs to be explicitly stepped

		# Add ground plane and set lateral friction coefficient
		self.groundId = self.client.loadURDF('plane.urdf') #: int: Body ID of ground
		self.client.changeDynamics(self.groundId, -1, lateralFriction=self.foot_friction)
		# Add hexapod URDF
		position = [0, 0, self.controller.body_height]
		orientation = self.client.getQuaternionFromEuler([0, 0, -controller.crab_angle])
		filepath = os.path.abspath(os.path.dirname(__file__)) + urdf
		self.hexId = self.client.loadURDF(filepath, position, orientation, flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION) #: int: Body ID of hexapod robot
		# get joint and link info from model
		self.joints = self.__get_joints(self.hexId) #: list of int: List of joint indeces. These are the angles of the servos
		self.links = self.__get_links(self.joints) #: list of int: List of link indeces
		# initialise joints and links
		self.__init_joints(self.controller, self.joints, self.locked_joints)
		self.__init_links(self.links)

		# self.fps = 24
		# self.pipeline = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'mjpeg', '-r', str(self.fps), '-i', '-', '-vcodec', 'mpeg4', '-qscale', '5', '-r', str(self.fps), 'video.mp4'], stdin=PIPE)


	# set joints to their initial positions
	def __init_joints(self, controller, joints, locked_joints):
		joint_angles = controller.joint_angles(t=0)
		for index, joint in enumerate(joints):
			joint_angle = joint_angles[index]
			joint_index, lower_limit, upper_limit, max_torque, max_speed = joint
			# not guaranteed that joint is present
			if joint_index is None: continue
			# if joint is locked, set it to fixed angle
			if index in locked_joints:
				joint_speed = 0
				if joint_index in np.arange(18)[0::3]: # coxa
					joint_angle = np.radians(0)
				elif joint_index in joints[1::3]: # femur
					joint_angle = np.radians(90)
				elif joint_index in joints[2::3]: # tibia
					joint_angle = np.radians(-150)
			# set joints to their starting position
			self.client.resetJointState(self.hexId, joint_index, targetValue=joint_angle)
			if index in locked_joints: continue
			# ensure actuator behaves as unpowered servo
			# assign small friction force to joint to simulate servo friction
			self.client.setJointMotorControl2(self.hexId, joint_index, p.VELOCITY_CONTROL, force=0.1)


	def __init_links(self, links):
		tibia_links = links[:, 2]
		for link_index in tibia_links:
			# assign friction to feet to ensure they are not dragged
			self.client.changeDynamics(self.hexId, link_index, lateralFriction=self.foot_friction)
		# remove collisions between femur and base as this was preventing full coxa range of motion
		femur_links = links[:, 1]
		for link_index in femur_links:
			self.client.setCollisionFilterPair(self.hexId, self.hexId, linkIndexA=-1, linkIndexB=link_index, enableCollision=0)


	# Fetches and stores the joint index and joint information in the expected order
	def __get_joints(self, robotId):
		# A lot of the joint information in the URDF is not used in setJointMotorControl and needs to be manually applied
		joint_names = [b'joint_1_1', b'joint_1_2', b'joint_1_3', b'joint_2_1', b'joint_2_2', b'joint_2_3', b'joint_3_1', b'joint_3_2', b'joint_3_3', b'joint_4_1', b'joint_4_2', b'joint_4_3', b'joint_5_1', b'joint_5_2', b'joint_5_3', b'joint_6_1', b'joint_6_2', b'joint_6_3']
		joints = np.full((len(joint_names), 5), None)
		for joint_index in range(self.client.getNumJoints(robotId)):
			info = self.client.getJointInfo(robotId, joint_index)
			try:
				index = joint_names.index(info[1])
				# [ joint_index, lower_limit, upper_limit, max_torque, max_velocity ]
				joints[index] = [info[0], info[8], info[9], info[10], info[11]]
			except ValueError:
				print('Unexpected joint name in URDF')
		return joints


	def __get_links(self, joints):
		# In pybullet the linkIndex is the jointIndex
		link_indices = self.joints[:,0]
		links = link_indices.reshape(6,3)
		return links


	def set_foot_friction(self, foot_friction):
		self.foot_friction = foot_friction
		self.client.changeDynamics(self.groundId, -1, lateralFriction=foot_friction)


	def terminate(self):
		"""Closes PyBullet physics engine and frees up system resources

        Note
        ----
        Prints the PyBullet error if termination failed

        """
		try:
			# self.client.stopStateLogging(self.logId)
			# self.pipeline.stdin.close()
			# self.pipeline.wait()
			self.client.disconnect()
		except p.error as e:
			print('Termination of simulation failed:', e)


	def step(self):
		"""Steps the simulation by dt

        Note
        ----
        This will request the joint angles and speeds from the assigned controller

        """
		returnDifference = 0
		start_time = time.perf_counter()
		if self.count == 0:
			self.count += 1
		# using setJointMotorControl2 (slightly slower but allows setting of max velocity)
		joint_angles = self.controller.joint_angles(t=self.t)
		if self.count == 0:
			self.count += 1
		else:
			difference = [abs(x-y) for x, y in zip(joint_angles, self.previous_joint_angles)]
			returnDifference = sum(difference)

		for index, joint_properties in enumerate(self.joints):
			joint_index, lower_limit, upper_limit, max_torque, max_speed = joint_properties
			
			# skip if joint is not present of if failed
			if (joint_index is None) or (index in self.locked_joints) or (index in self.failed_joints): continue

			joint_angle = joint_angles[index]
			
			# ensure controller does not attempt to exceed joint limits
			joint_angle = min(max(lower_limit, joint_angle), upper_limit)

			# max velocity in URDF isn't used and needs to be assigned
			self.client.setJointMotorControl2(self.hexId, joint_index, p.POSITION_CONTROL, targetPosition=joint_angle, force=max_torque, maxVelocity=max_speed)

		if self.collision_fatal:
			if self.__link_collision() or self.__ground_collision():
				raise RuntimeError('Link collision during simulation')

		end_time = time.perf_counter()
		# follow robot with camera
		if self.visualiser_enabled:
			if self.follow:
				self.client.resetDebugVisualizerCamera(cameraDistance=self.camera_distance, cameraYaw=self.camera_yaw, cameraPitch=self.camera_pitch, cameraTargetPosition=self.base_pos())
			time.sleep(max(self.dt - end_time + start_time, 0))

		# controller feedback running at 10Hz
		if ((self.n_step % 24) == 0):
			self.controller.IMU_feedback(self.base_orientation())

		self.client.stepSimulation()
		self.n_step += 1
		self.t += self.dt
		self.previous_joint_angles = joint_angles
		return returnDifference
	
	def supporting_legs(self):
		"""Determines the supporting legs for the hexapod

		Note
		----
		A leg is considered to be supporting if it is in constact with the ground.
		This method is used in plotting the gait sequence diagram.

        Returns
        ----
        list of bool: A list of booleans where the index is the leg number and where 'True' represents in contact with the ground.

        """
		tibia_links = self.links[:, 2]
		# Get contact points between hex and ground for the last stepSimulation call
		contact_points = np.array(self.client.getContactPoints(self.hexId, self.groundId), dtype=object)
		# get links for contact between ground and hexapod
		try:
			contact_links = contact_points[:, 3]
		except IndexError as e:
			contact_links = np.array([])
		supporting_legs = np.isin(tibia_links, contact_links)

		return supporting_legs


	# Returns true for collision between links in robot
	def __link_collision(self):
		# Get contact points between hex links
		# Collision between child and parent in URDF is disabled by default so shouldn't need to ignore this
		contact_points = np.asarray(self.client.getContactPoints(self.hexId, self.hexId), dtype=object)
		# also need to add in collision detection between robot and ground excluding tibia links
		return contact_points.size > 0


	# Returns true for collisions between the robot and the ground
	def __ground_collision(self):
		tibia_links = self.links[:, 2]
		# Get contact points between hex and ground for the last stepSimulation call
		contact_points = np.array(self.client.getContactPoints(self.hexId, self.groundId), dtype=object)
		# get links for contact between ground and hexapod excluding the tibia links as these can contact ground
		try:
			contact_links = contact_points[:, 3]
		except IndexError as e:
			contact_links = np.array([])
		# filter out tibia contacts
		contact_links = contact_links[~np.isin(contact_links, tibia_links)]

		# returns a boolean array of whether tibia is in contact with the ground or not
		return contact_links.size > 0


	def base_orientation(self):
		quaternion = self.client.getBasePositionAndOrientation(self.hexId)[1]
		return p.getEulerFromQuaternion(quaternion)


	# returns the base position without the orientation
	def base_pos(self):
		"""Returns the position of the hexapod base

		Note
		----
		The base orientation is not returned

        Returns
        ----
        list of float: The position of the hexapod base in cartesian coordinates

        """
		return self.client.getBasePositionAndOrientation(self.hexId)[0]


def evaulate_gait(leg_params, body_height=0.14, velocity=0.3, duration=5.0, visualiser=True, collisions=False, failed_legs=[]):
	# controller will return an error if parameters are not feasible
	controller = Controller(leg_params, body_height=body_height, velocity=velocity, crab_angle=-np.pi/6)
	# initialise simulator
	simulator = Simulator(controller, follow=True, visualiser=visualiser, collision_fatal=collisions, failed_legs=failed_legs)
	# initialise fitness and descriptor
	contact_sequence = np.full((6, 0), False)
	# simulator returns error if collision occurs
	for t in np.arange(0, duration, step=simulator.dt):
		try:
			simulator.step()
		except RuntimeError as error:
			print(error)
			fitness = 0
			break
		contact_sequence = np.append(contact_sequence, simulator.supporting_legs().reshape(-1,1), axis=1)
	fitness = simulator.base_pos()[0]
	# summarise descriptor
	descriptor = np.sum(contact_sequence, axis=1) / np.size(contact_sequence, axis=1)
	# plot footfall diagram
	plot_footfall(contact_sequence)

	simulator.terminate()

	return fitness, descriptor




if __name__ == "__main__":
	from controllers.kinematic import Controller
	from controllers.kinematic import stationary
	
	controller = Controller(stationary, body_height=0.11, velocity=0.0, crab_angle=-np.pi/6)
	simulator = Simulator(controller=controller, follow=False, visualiser=True, collision_fatal=False, camera_distance=1.0, camera_yaw=90, camera_pitch=-55)
	while True:
		simulator.step()
