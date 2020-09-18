import numpy as np
from scipy.spatial.transform import Rotation as R

from dVRL_simulator.PsmEnv import PSMEnv
from dVRL_simulator.vrep.simObjects import table, rail, targetK, collisionCheck #,obj, target
from dVRL_simulator.vrep.vrepObject import vrepObject

import transforms3d.euler as euler
import transforms3d.quaternions as quaternions


import time



def goal_distance(goal_a, goal_b):
	assert goal_a.shape == goal_b.shape
	return np.linalg.norm(goal_a - goal_b, axis=-1)


class PSMEnv_Position(PSMEnv):

	def __init__(self, psm_num, n_substeps, block_gripper,
				has_object, target_in_the_air, height_offset, target_offset, obj_range, target_range,
				distance_threshold, initial_pos, reward_type, dynamics_enabled, two_dimension_only,
				randomize_initial_pos_obj, randomize_initial_pos_ee, docker_container,
				randomize_initial_or_obj, randomize_initial_pos_kidney, randomize_initial_or_kidney):

		"""Initializes a new signle PSM Position Controlled Environment
		Args:
			psm_num (int): which psm you are using (1 or 2)
			n_substeps (int): number of substeps the simulation runs on every call to step
			gripper_extra_height (float): additional height above the table when positioning the gripper
			block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
			has_object (boolean): whether or not the environment has an object
			target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
			height_offset (float): offset from the table for everything
			target_offset ( array with 3 elements): offset of the target, usually z is set to the height of the object
			obj_range (float): range of a uniform distribution for sampling initial object positions
			target_range (float): range of a uniform distribution for sampling a target Note: target_range must be set > obj_range
			distance_threshold (float): the threshold after which a goal is considered achieved
			initial_pos  (3x1 float array): The initial position for the PSM when reseting the environment. 
			reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
			dynamics_enabled (boolean): To enable dynamics or not
			two_dimension_only (boolean): To only do table top or not. target_in_the_air must be set off too.
			randomize_initial_pos_obj (boolean)
			docker_container (string): name of the docker container that loads the v-rep
			randomize_initial_or_obj (boolean)
			randomize_initial_pos_kidney (boolean)
			randomize_initial_or_kidney (boolean)

		"""
		#self.gripper_extra_height = gripper_extra_height
		self.block_gripper = block_gripper
		self.has_object = has_object
		self.target_in_the_air = target_in_the_air
		self.height_offset = height_offset
		self.target_offset = target_offset
		self.obj_range = obj_range
		self.target_range = target_range
		self.distance_threshold = distance_threshold
		self.initial_pos = initial_pos
		self.reward_type = reward_type
		self.dynamics_enabled = dynamics_enabled
		self.two_dimension_only = two_dimension_only
		self.randomize_initial_pos_obj = randomize_initial_pos_obj
		self.randomize_initial_pos_ee = randomize_initial_pos_ee
		self.randomize_initial_or_obj = randomize_initial_or_obj
		self.randomize_initial_pos_kidney = randomize_initial_pos_kidney
		self.randomize_initial_or_kidney = randomize_initial_or_kidney




		if self.block_gripper:
			self.n_actions = 3 
			self.n_states  = 3 + self.has_object*3 
		else:
			self.n_actions = 4
			self.n_states  = 4 + self.has_object*3


		super(PSMEnv_Position, self).__init__(psm_num = psm_num, n_substeps=n_substeps, n_states = self.n_states, 
						      n_goals = 3, n_actions=self.n_actions, camera_enabled = False,
						      docker_container =docker_container)


		#self.target = target(self.clientID, psm_num)
		self.targetK = targetK(self.clientID)
		self.collisionCheck = collisionCheck(self.clientID, psm_num)

		self.vrepObject=vrepObject(self.clientID)

		if self.has_object:
			#self.obj = obj(self.clientID)
			self.rail = rail(self.clientID)
		self.table = table(self.clientID)

		self.prev_ee_pos  = np.zeros((3,))
		self.prev_ee_rot  = np.zeros((3,))
		self.prev_obj_pos = np.zeros((3,))
		self.prev_obj_rot = np.zeros((3,))
		self.prev_jaw_pos = 0

		if(psm_num == 1):
			self.psm = self.psm1
		else:
			self.psm = self.psm2


		#Start the streaming from VREP for specific data:

		#PSM Arms:
		self.psm.getPoseAtEE(ignoreError = True, initialize = True)
		self.psm.getJawAngle(ignoreError = True, initialize = True)
		
		#Used for _sample_goal
		#self.target.getPosition(self.psm.base_handle, ignoreError = True, initialize = True)
		self.targetK.getPosition(self.psm.base_handle, ignoreError = True, initialize = True)
		

		#Used for _reset_sim
		self.table.getPose(self.psm.base_handle, ignoreError = True, initialize = True)
		if self.has_object:
			#self.obj.getPose(self.psm.base_handle, ignoreError = True, initialize = True) #Also used in _get_obs
			self.rail.getPose(self.rail.dummy1_rail_handle, self.psm.base_handle, ignoreError = True, initialize = True)
			self.rail.getPose(self.rail.dummy2_rail_handle, self.psm.base_handle, ignoreError = True, initialize = True)
			self.rail.getPose(self.rail.dummy3_rail_handle, self.psm.base_handle, ignoreError = True, initialize = True)
			self.rail.getPose(self.rail.dummy4_rail_handle, self.psm.base_handle, ignoreError = True, initialize = True)
			self.rail.getPose(self.rail.dummy5_rail_handle, self.psm.base_handle, ignoreError = True, initialize = True)
			self.rail.getPose(self.rail.dummy6_rail_handle, self.psm.base_handle, ignoreError = True, initialize = True)
			self.rail.getPose(self.rail.dummy7_rail_handle, self.psm.base_handle, ignoreError = True, initialize = True)
			self.rail.getPose(self.rail.dummy8_rail_handle, self.psm.base_handle, ignoreError = True, initialize = True)
			self.rail.getPose(self.rail.rail_handle, self.psm.base_handle, ignoreError = True, initialize = True)
			#self.rail.getPose(self.rail.rail_res_handle, self.psm.base_handle, ignoreError = True, initialize = True)
			self.rail.getPose(self.rail.rail_achieved_top, self.psm.base_handle, ignoreError = True, initialize = True)
			self.rail.getPose(self.rail.rail_achieved_bottom, self.psm.base_handle, ignoreError = True, initialize = True)
			self.rail.getPose(self.rail.rail_achieved_central, self.psm.base_handle, ignoreError = True, initialize = True)

			#Used for _get_obs
			#self.obj.isGrasped(ignoreError = True, initialize = True)
			self.rail.isGrasped(ignoreError=True, initialize=True)


	# GoalEnv methods
	# ----------------------------

	def compute_reward(self, achieved_goal, goal, info):

		d = goal_distance(achieved_goal, goal)*self.target_range #Need to scale it back!

		if self.reward_type == 'sparse':
			return -(d > self.distance_threshold).astype(np.float32)
		else:
			return -100*d

	# PsmEnv methods
	# ----------------------------


	def _set_action(self, action#, new
			):
		'''
		@Info: 
		Set the EE to a new position and orientation, closer to the targets' ones, at each time-step.
		
		- 1st step: copy action from script/policy
		
		- 2nd step: if the LND is closed, then gripper_ctrl is 0. The action is simply position_ctrl.
		Else, the gripper_ctrl is the fourth element of the action vector.

		- 3rd step: pos_ctrl is multiplied by 1 mm, which is the eta factor discussed in dVRL paper.

		- 4th step: check if the toop-tip of the EE is below the table. If it is, raise it above the table.

		- Check step: if the rail is not grasped, we are in the pick phase. Therefore, we execute the code
		for the pick phase. The EE gets closer to the grasping site and orientates perpendicular to the rail.
		Else, if the rail is grasped, the EE moves towards the kidney and orientates to lay flat the rail on it.
		Like this:
			Grasped=True step: get a new quaternion for the EE, closer to the target's orientation. Set it to the new
			quaternion if the orientations are not yet close enough (threshold dictates this). Else, set the orientation
			equal to the target's. This is done because the error doesn't converge to 0, due to the instability
			of setting an orientation in V-Rep. 
		'''
		assert action.shape == (self.n_actions,)
		action = action.copy()  #ensure that we don't change the action outside of this scope

		if self.block_gripper:
			pos_ctrl = action[0:3]
			gripper_ctrl = 0
		else:
			pos_ctrl, gripper_ctrl = action[0:3], action[3]
			gripper_ctrl = (gripper_ctrl+1.0)/2.0 #gripper_ctrl bound to 0 and 1

		grasped = self.rail.isGrasped()

		#Get EE's pose:
		pos_ee, quat_ee = self.psm.getPoseAtEE()
		#Add position control:
		pos_ee = pos_ee + pos_ctrl*0.001 #as the paper states, eta = 1mm used to avoid overshoot on real robot

		#Get table information to constrain orientation and position:
		pos_table, q_table = self.table.getPose(self.psm.base_handle)
		#Make sure tool tip is not in the table by checking tt and which side of the table it is on.
		#DH parameters to find tt position:
		ct = np.cos(0)
		st = np.sin(0)

		ca = np.cos(-np.pi/2.0)
		sa = np.sin(-np.pi/2.0)

		T_x = np.array([[1,  0,  0, 0],
		               [0, ca, -sa, 0],
		               [0, sa,  ca, 0],
		               [0, 0, 0,    1]])
		T_z = np.array([[ct, -st, 0, 0],
		                [st,  ct, 0, 0],
		                [0,    0, 1, 0.0102],
		                [0,    0, 0, 1]])

		ee_T_tt = np.dot(T_x, T_z)

		pos_tt, quat_tt = self.psm.matrix2posquat(np.dot(self.psm.posquat2Matrix(pos_ee, quat_ee), ee_T_tt))

		pos_tt_on_table, distanceFromTable = self._project_point_on_table(pos_tt)

		# If the distance from the table is negative, then we need to project pos_tt onto the table top.
		# Or if two dim only are enabled.
		if distanceFromTable < 0 or self.two_dimension_only:
			pos_ee, _ = self.psm.matrix2posquat(np.dot(self.psm.posquat2Matrix(pos_tt_on_table, quat_tt), np.linalg.inv(ee_T_tt)))

		#Make sure the new pos doesn't go out of bounds!!!
		#Note: these are the bounds for the reachable space of the EE.
		upper_bound = self.initial_pos + self.target_range + 0.01
		lower_bound = self.initial_pos - self.target_range - 0.01

		pos_ee = np.clip(pos_ee, lower_bound, upper_bound)
		
		q_target = self.targetK.getOrientationGoals(self.psm.base_handle)

		if not grasped:
			#If rail not grasped, change orientation of EE to rail's.
			_, q_dummy = self.rail.getPose(self.dummy_rail_handle, self.psm.base_handle)
			temp_q = quaternions.qmult([q_dummy[3], q_dummy[0], q_dummy[1], q_dummy[2]], [0.7, -0.7, 0, 0])
			new_ee_quat = np.array([temp_q[1], temp_q[2], temp_q[3], temp_q[0]])
			self.psm.setPoseAtEE(pos_ee, new_ee_quat, gripper_ctrl)
		if grasped:
			#If rail is grasped, change slowly orientation of EE with the rail grasped to kidney's.
			new_ee_quat, done = self._align_to_target()
			if done: #if the EE (and rail) is oriented like the target, stop changing orientation
				#self.psm.setPoseAtEE(pos_ee, quat_ee, gripper_ctrl)
				self.psm.setPoseAtEE(pos_ee, q_target, gripper_ctrl)
			else:
				self.psm.setPoseAtEE(pos_ee, new_ee_quat, gripper_ctrl)


		#print("New before align", new)
		#Neri: compute orientation control
		#new_ee_quat, done, new = self._align_to_target(new)
		#print("New after align, should equal new degrees EE:", new)

		#return done


	def _align_to_target(self, k = 0.15, threshold = 8 #, new
				):
		'''
		@Info:
		This method is used by _set_action() to compute the error in orientation
		between the EE and the target when Grasped = True.
		The error is computed by subtracting components values one by one.

		- 1st step: we get the quaternions of the frames of target and EE. 
		@Note: the target's quaternion is not the cuboid's, but the quaternion of a dummy
		centered in the cuboid called "Kidney_orientation_ctrl". Why? This has the y axis downward,
		so I can compute the orientation error in a more uncomplicated way, just subtracting corresponding
		components.

		- 2nd step: from these quaternions we obtain angles in degrees so we can
		compute the error between the orientations.
		@Note: these two phases should be done by premultiplying the transposed rotation matrix
		of EE by the rotation matrix of target. This is the orientation error matrix between the two frames.
		From this, we obtain the angles in degrees (remember constraints on angles!).

		- 3rd step: we compute the errors and move the EE orientation by 10% of the error. 
		X and Z angles however also have a proportional factor "k" to increase stability. 
		This factor was chosen empirically.

		- 4th step: get the new quaternion of the EE and return it to _set_action() to set pose at EE.

		- 5th step: if the norm of the error vector is less than value "threshold" (empirically set), then
		we consider the orientation reached, done = True.

		@Returns: new_quaternion for EE, done signal.
		'''
		#Get pose of target:
		q_target = self.targetK.getOrientationGoals(self.psm.base_handle)
		#Convert target quaternion to euler angles (radians):
		eul_target = self.vrepObject.quat2Euler(q_target)
		#Convert target euler angles to degrees:
		eul_target_deg = eul_target*(180/np.pi)

		#Get pose of EE:
		_, q_ee = self.psm.getPoseAtEE()
		#Convert EE quaternion to euler angles (radians)
		eul_ee = self.vrepObject.quat2Euler(q_ee)
		#Convert EE euler angles to degrees:
		eul_ee_deg = eul_ee*(180/np.pi) # (-90,-20,0) (rxyz)

		#Sort of proportional control. Due to the instability of setting the orientation of the EE,
		#the quaternion we want the EE to go to is different from the quaternion set by V-Rep!
		#Therefore, errors sum up causing a non perfect alignment of the frames. 
		#To limit this, this parameter k was used to do some sort of proportional control. 
		#Not applied to error y because it caused more issues.
		delta_rot_x = eul_target_deg[0] - eul_ee_deg[0] -k*(eul_target_deg[0] - eul_ee_deg[0])
		delta_rot_y = eul_target_deg[1] - eul_ee_deg[1]
		delta_rot_z = eul_target_deg[2] - eul_ee_deg[2] -k*(eul_target_deg[2]-eul_ee_deg[2])

		#We want to slowly reach the target's orientation.
		#At each time-step, the EE is rotated by 10% the delta_rot at that time-step.
		rot_ctrl_x = delta_rot_x * 0.1
		rot_ctrl_y = delta_rot_y * 0.1
		rot_ctrl_z = delta_rot_z * 0.1

		#The new orientation for the EE is its previous + the change in orientation along each axis:
		new_eul_ee_deg = np.array([eul_ee_deg[0]+rot_ctrl_x, eul_ee_deg[1]+rot_ctrl_y, eul_ee_deg[2]+rot_ctrl_z])		
		#Back to radians:
		new_eul_ee = new_eul_ee_deg*(np.pi/180)
		#Back to quat:
		new_q_ee = self.vrepObject.euler2Quat(new_eul_ee)

		#NOTE: 
		#Print the new quaternion, which we'll use to set the pose of the EE, then check the "old" quaternion
		#at the next time-step. You'll see they are different, because the setPose method has numerical errors.

		done = False
		#If the orientation is almost the one of the target, stop adding the difference:
		norm_delta_rot = np.linalg.norm(np.array([delta_rot_x,delta_rot_y,delta_rot_z])) #"almost" is quantified by the norm of the error vector
		if norm_delta_rot < threshold:
			done = True
		else:
			done = False

		return new_q_ee, done #, new_eul_ee_deg

		#NOTE: the best would be not to read the EE quaternion to calculate the current error
		#at a given time-step, but employ the new_eul_ee_deg to compute the next error (use the desired, not actual orientation).
		#However, I couldn't understand how to make this in this script, because
		#at any time-step, these methods are executed top to bottom. So, how can I save its value 
		#and read it at the next time-step to compute the error with it? I tried one implementation but didn't work.
		#Variable "new" is part of this attempt.


	def _get_obs(self):
		'''
		@Info:
		This get obs method builds the dict comprised of observation,
		achieved goal and desired goal. Desired goal is always the central dummy
		on the kidney. This dummy is one of the 5 possible dummies and randomised. 

		Achieved goal and observation change according to whether the rail is grasped or not.
		If the rail is not grasped: the goal is to reach the grasping site of the rail and pick the rail.
		If the rail is grasped: the goal is to reach the kidney and place the rail. 
		'''
		#Normalize ee_position:
		ee_pos,  _ = self.psm.getPoseAtEE()
		ee_pos = (ee_pos - self.initial_pos)/self.target_range

		jaw_pos = self.psm.getJawAngle()

		if self.has_object:
			grasped = self.rail.isGrasped()
			if not grasped:
				#If not grasped, object used to position control is grasping site
				obj_pos,  _ = self.rail.getPose(self.dummy_rail_handle, self.psm.base_handle)
				obj_pos = (obj_pos - self.initial_pos)/self.target_range
				#Achieved goal is pos of grasping site
				achieved_goal = np.squeeze(obj_pos)
			if grasped: 
				#If grasped, the object is the central dummy below the rail, also goal
				achieved_goal_central, _, _ = self.rail.getPositionAchievedGoals(self.psm.base_handle)
				obj_pos = (achieved_goal_central - self.initial_pos)/self.target_range
				achieved_goal = np.squeeze(obj_pos)

			obs = np.concatenate((ee_pos, np.array([jaw_pos]), obj_pos)) 

		else:
			obj_pos = np.zeros((3,))
			achieved_goal = np.squeeze(ee_pos)
			if self.block_gripper:
				obs = ee_pos
			else:
				obs = np.concatenate((ee_pos, np.array([jaw_pos]))) 		

		self.prev_ee_pos  = ee_pos
		self.prev_ee_rot  = np.zeros((3,))
		self.prev_obj_pos = obj_pos
		self.prev_obj_rot = np.zeros((3,))
		self.prev_jaw_pos = jaw_pos


		return {
				'observation': obs.copy(),
				'achieved_goal': achieved_goal.copy(),
				'desired_goal' : self.goal.copy()
		}


	def _reset_sim(self, initial_tt_offset=0.035, initial_rail_offset=0.015, minimum_d=0.015):
		'''
		@Info: this method sets the EE pose and the Rail's pose.
		'''
		
		#Get the constrained orientation of the ee
		pos_table, q_table = self.table.getPose(self.psm.base_handle)

		temp_q =  quaternions.qmult([q_table[3], q_table[0], q_table[1], q_table[2]], [ 0.5, -0.5, -0.5,  0.5])
		ee_quat_constrained = np.array([temp_q[1], temp_q[2], temp_q[3], temp_q[0]])

		#Put the EE in the correct orientation
		self.psm.setDynamicsMode(0, ignoreError = True)
		self._simulator_step

		if self.randomize_initial_pos_ee:

			if self.target_in_the_air:
				z = self.np_random.uniform(0, self.obj_range) + initial_tt_offset
			else:
				z = initial_tt_offset

			#Add target_offset for goal. 
			random_EE_pos = np.append(self.np_random.uniform(-self.obj_range, self.obj_range, size=2), [z])

			#Project EE on to the table and add the deltaEEPos to that
			pos_ee_projectedOnTable,_ = self._project_point_on_table(self.initial_pos)
			pos_ee = pos_ee_projectedOnTable + random_EE_pos

		else:
			pos_ee = self.initial_pos

		self.psm.setPoseAtEE(pos_ee, ee_quat_constrained, 0, ignoreError = True)

		if self.has_object:
			#self.obj.removeGrasped(ignoreError = True)
			self.rail.removeGrasped(ignoreError = True)
		self._simulator_step

		if self.dynamics_enabled:
			self.psm.setDynamicsMode(1, ignoreError = True)


		if self.has_object: #When re-writing the whole code because of Dynamics not working, I didn't re-write all Claudia's methods.
					#but this does the same thing, except printing grasping site color and number.

			z = initial_rail_offset
			
			dist_from_ee = 0
			while dist_from_ee < minimum_d:

				if self.randomize_initial_pos_obj:
					x = self.np_random.uniform(-self.obj_range, self.obj_range)
					y = self.np_random.uniform(-self.obj_range, self.obj_range)
				else:
					x = 0
					y = 0

				random_obj_pos = np.array([x,y,z])

				#Project initial EE on to the table and add the deltaObject to that
				pos_ee_projectedOnTable,_ = self._project_point_on_table(self.initial_pos)
				obj_pos = pos_ee_projectedOnTable + random_obj_pos

				if self.randomize_initial_pos_obj:
					dist_from_ee = np.linalg.norm(obj_pos - pos_ee)
				else:
					dist_from_ee = 1

			#self.obj.setPose(obj_pos, q_table, self.psm.base_handle, ignoreError = True)

			if self.randomize_initial_or_obj:
				q_dummy = self._randomize_dummy_orientation()
			else:
				q_dummy = q_table
			grasp_site = self.np_random.randint(1, 9)
			self.dummy_rail_handle, rail_pos = self.rail.setPose(obj_pos, q_dummy, grasp_site, self.psm.base_handle, ignoreError=True)

			self.prev_obj_pos = obj_pos
			self.prev_obj_rot = self.psm.quat2Euler(q_table)
	
			
		else:
			self.prev_obj_pos = self.prev_obj_rot = np.zeros((3,))

		self.prev_ee_pos  = pos_ee
		self.prev_ee_rot  = self.psm.quat2Euler(ee_quat_constrained)
		self.prev_jaw_pos = 0
		
		self._simulator_step()
		self._correct_rail_pos(q_dummy)


		return True

	def _randomize_dummy_orientation(self):
                '''
                @Info: Claudia's method for setting a random orientation for the grasping site.
                '''
                x = self.np_random.randint(-30, 0)
                y = self.np_random.randint(-45, 45)
                z = self.np_random.randint(-45, 45)

                # The orientation of the rail is defined in the /base which is why we 
                # follow order [x, z, y]
                rot_dummy = R.from_euler('yxz', [y, x, z], degrees=True)
                q_dummy = rot_dummy.as_quat()        
                return q_dummy

	def _correct_rail_pos(self, q_dummy, grasp_site=8, safe_offset = [0, 0, 0.008]):
                '''
                @Info: Claudia's method for correcting rail's pose if under the table.
                '''

                # Get the position of the dummy8 in the /base
                pos_dummy_eight, _ = self.rail.getPose(self.rail.dummy8_rail_handle, self.psm.base_handle)

                # Project dummy8 position on the table and get distance between dummy8 
                # and table surface
                pos_dummy_eight_on_table, distanceFromTable_eight = self._project_point_on_table(pos_dummy_eight)

                # Check if the dummy8 is below the table
                if distanceFromTable_eight < 0:
                        # Move dummy8 above the table
                        pos_dummy_eight_above_table = pos_dummy_eight_on_table + safe_offset

                        # Move the rail above the table
                        _ , pos_rail_set  = self.rail.setPose(pos_dummy_eight_above_table, q_dummy, grasp_site, self.psm.base_handle, ignoreError=True)
                        self._simulator_step()
                        print('Position of the rail has been corrected')


	#Must be called immediately after _reset_sim since the goal is sampled around the position of the EE
	def _sample_goal(self):
		'''
		@Info: 
		Once the pose is set in randomize_k_pos, a random set of targets is sampled.
		There are 5 triplets of dummies on the cuboid's surface. They are one central, one bottom, one top.
		The position of the central target is set as desired_goal. The rail will have to be laid on the kidney
		so that a central dummy below it, called achieved_goal_central, reaches the central dummy on the cuboid.

		@Returns: goal.copy, the position of the central target on the cuboid.
		
		'''
		self._simulator_step()

		self.randomize_k_pos()

		#Once the pose is set, sample a target off the 5 sets. The goal is only the central dummy.
		#The top and bottom targets are not useful here. Later, in is_success, they will be used
		#to check if they correspond to the targets below the rail (orange dummies), achieved_goal_top and achieved_goal_bottom.
		goal = self.targetK.getPositionGoal(self.psm.base_handle)
		goal = (goal - self.initial_pos)/self.target_range
 
		return goal.copy()


	def randomize_k_pos(self, initial_pos_k = [0.05, -0.05, 0]):
		'''
		@Info: 
		Randomisation of the kidney is done if randomize_initial_pos_kidney is True in config file.
		Randomisation is done this way: x and y are sampled in ranges = +- 5cm (2*obj_range)
		z = 0 + target_offset = [0,0,38] mm
		Then we add the vertical translation of pos_ee_projectedOnTable

		Orientation is sampled by method randomize_k_orientation if desired

		The pose of the kidney is randomised like this until no collisions are registered by the method
		contained in simObject called KidneyCollision. 

		If this isn't done in 100 tries, then the simulation is reset, meaning the EE and Rail 
		have new random poses and we try again for 100 tries in this new configuration.

		If randomisation is not desired, a standard fixed pose is set. Position is initial_pos_k in input.
		Orientation is below. Note that this doesn't collide with Rail nor EE if also they are set to their non-random fixed poses.
		'''
		z = 0
		collision = True 
		i = 1
		j = 1
		pos_ee_projectedOnTable, _ = self._project_point_on_table(self.initial_pos)
		randomize = self.randomize_initial_pos_kidney
        
		#If you want the position of the kidney randomised:
		if randomize:
			#Until things don't collide:
			while collision == True:
				# Step 1: random x,y coordinates and fixed z.
				random_kidney_pos = np.append(self.np_random.uniform(-2*self.obj_range, 2*self.obj_range, size=2), [z]) + self.target_offset
				# Project initial_pos on the table-top and add the deltaGoal to that,
				# therefore performing a vertical translation of the (x,y,z) coordinates
				# towards the table.
				kidney_pos = pos_ee_projectedOnTable + random_kidney_pos
				if self.randomize_initial_or_kidney:
					rand_cuboid_quat = self.randomize_k_orientation()
				else:
					rand_cuboid_quat = [0, 0, 1, 0]

				# Step 2: set the pose with the above position and quaternion.
				self.targetK.setPose(kidney_pos, rand_cuboid_quat, self.psm.base_handle, ignoreError=True)
				self._simulator_step()

				# Check if the just set pose causes any collision between the kidney
				# and other inputed shapes. c_r = collision result.
				c_r1, c_r2, c_r3, c_r4, c_r5, c_r6, c_r7 = self.collisionCheck.KidneyCollision()
				# If it doesn't collide with anything, so every collision state is False, perfect!
				# Else, repeat and randomise again.
				if (not c_r1 and not c_r2 and not c_r3 and not c_r4 and not c_r5 and not c_r6 and not c_r7):
					collision = False
				else: 
					i = i + 1
				#print("Computing new pose, try number", i)
					collision = True
				# If you can't find any good pose in 100 tries, try resetting!
				if i == 100*j:
					j = j+1
					print("RESET number:", j-1)
					self._reset_sim()

			if collision == True:
				print("Colliding objects.")
			else: 
				print("Objects shouldn't be colliding.")

		# Else, set the cuboid to a fixed non-colliding position. Note that it
		# will not collide with the rail nor the robot if they are set to their own
		# fixed positions too (all randomisation booleans must be False).
		else:
			kidney_pos = np.array(initial_pos_k) + np.array(self.target_offset) + np.array(pos_ee_projectedOnTable)
			# These rot angles are ok to avoid collision
			x_rot = 0
			y_rot = 0
			z_rot = -10
			rot = R.from_euler('yxz', [y_rot, x_rot, z_rot], degrees=True)
			fixed_quat = rot.as_quat()

			self.targetK.setPose(kidney_pos, fixed_quat, self.psm.base_handle, ignoreError=True)
			self._simulator_step()


	def randomize_k_orientation(self):
		'''
		@Info:
		Randomise the kidney's orientation.
		In kidney cuboid's frame: rotation ranges are defined by:
		- Pitch rotation, about the x axis between (-20,20)
		- Roll rotation, about the y axis between (-30,30)
		- Yaw rotation, about the z axis between (-89, 0) 

		@Note:
		The z angle is not between (-89,90) because of this:
		the rail has to reach the orientation of the kidney_orientation_ctrl dummy.
		Why: this dummy is oriented with y axis downward and x and z axes so that
		the difference between orientation components x, y, z can be computed in _align_to_target()
		easily, just subtracting corresponding components of angles of EE and kidney (target).
		However, due to the orientation of x and z, the rail is laid flat with the suction channel
		towards the opposite side of the kidney's adrenal gland. 
		Therefore, if you allow the kidney to have a yaw of 90° for example, the rail will have to 
		do a big rotation to lay itself on the kidney so that the suction channel is against the side of the
		kidney's adrenal gland side (btw, I don't care if it is towards or against).
		This big difference in rotation causes the gripper to lose the rail while trying to rotate that much.
		SO: I didn't have time to implement something like: if the difference in rotation is big
		lay the rail with the suction channel towards the adrenal gland. And decided to keep this angle between (-89,0).

		@NOTE: maybe this isn't due to the big orientation span to cover, but because I am working with the 
		inverse kinematics not yet fully adapted. Indeed, the IK need to allow the gripper to open, but atm
		it doesn't open, because I decided so together with Mario Selvaggio since with just one IK target, only
		half the gripper can open (so we decided to keep it close, but be able to orientate). 
		He is working on opening the gripper even with just one IK target. 

		@Returns: a random quaternion.
		'''
		#The rotation ranges are defined around the /cuboid
		x = self.np_random.randint(-20, 20) #Pitch
		y = self.np_random.randint(-30, 30) #Roll
		z = self.np_random.randint(-89, 0) #Yaw

		#Random orientation in radians:
		rand_eul = np.array([x,y,z])*(np.pi/180)
		#Random orientation as quaternion:
		rand_quat = self.vrepObject.euler2Quat(rand_eul)

		return rand_quat

	def _is_success(self, achieved_goal, desired_goal):
		#Achieved goal is a central dummy below the rail.
		#Desired goal is a central dummy on the kidney's surface.
		#Compute the distance between central dummy below the rail and central dummy on the surface of the kidney:
		d = goal_distance(achieved_goal, desired_goal)*self.target_range #Need to scale it back! 

		#Get the positions of the dummies below the rail, top and bottom:
		_, achieved_goal_t, achieved_goal_b = self.rail.getPositionAchievedGoals(self.psm.base_handle, ignoreError=True, initialize=True)
		#Get the positions of the dummies on the kidney's surface, top and bottom
		desired_goal_t, desired_goal_b = self.targetK.getPositionGoalTopBottom(self.psm.base_handle, ignoreError=True, initialize=True)

		#Compute the distance between top dummy below the rail and top dummy on the surface of the kidney:
		d_top = goal_distance(achieved_goal_t, desired_goal_t)*self.target_range #Need to scale it back!
		#Compute the distance between bottom dummy below the rail and bottom dummy on the surface of the kidney:
		d_bottom = goal_distance(achieved_goal_b, desired_goal_b)*self.target_range #Need to scale it back!

		#Return 1 only if all the distances are below the threshold.
		return (d < self.distance_threshold).astype(np.float32)*(d_top < self.distance_threshold).astype(np.float32)* \
									(d_bottom <self.distance_threshold).astype(np.float32)

	#Already accounts for height_offset!!!!
	def _project_point_on_table(self, point):
		pos_table, q_table = self.table.getPose(self.psm.base_handle)
		b_T_table = self.psm.posquat2Matrix(pos_table, q_table)

		normalVector_TableTop = b_T_table[0:3, 2]
		distanceFromTable = np.dot(normalVector_TableTop.transpose(), (point - ((self.height_offset)*normalVector_TableTop + pos_table)))
		point_projected_on_table = point - distanceFromTable*normalVector_TableTop

		return point_projected_on_table, distanceFromTable
