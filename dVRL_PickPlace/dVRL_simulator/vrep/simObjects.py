from dVRL_simulator.vrep.vrepObject import vrepObject
import numpy as np

class camera(vrepObject):
	def __init__(self, clientID, rgb = True):
		super(camera, self).__init__(clientID)
		self.camera_handle = self.getHandle('Vision_Sensor')
		self.rgb = rgb

		self.getVisionSensorImage(self.camera_handle, self.rgb, ignoreError = True, initialize = True)

	def getImage(self, ignoreError = False):
		data, resolution = self.getVisionSensorImage(self.camera_handle, self.rgb, ignoreError = ignoreError, 
													initialize = False)

		if self.rgb:
			return np.array(data, dtype=np.uint8).reshape([resolution[1], resolution[0], 3])
		else:
			return np.array(data, dtype=np.uint8).reshape([resolution[1], resolution[0]])


class table(vrepObject):
	def __init__(self, clientID):
		super(table, self).__init__(clientID)
		self.table_top_handle = self.getHandle('customizableTable_tableTop')


	def getPose(self, relative_handle, ignoreError = False, initialize = False):
		return self.getPoseAtHandle(self.table_top_handle, relative_handle, ignoreError, initialize)


class target(vrepObject):
	def __init__(self,clientID, psm_number):
		super(target, self).__init__(clientID)

		self.target_handle = self.getHandle('Target_PSM{}'.format(psm_number))

		self.getPosition(-1, ignoreError = True, initialize = True)

	def setPosition(self, pos, relative_handle, ignoreError = False):
		self.setPoseAtHandle(self.target_handle, relative_handle, pos, [1,0,0,1], ignoreError)

	def getPosition(self, relative_handle, ignoreError = False, initialize = False):
		pos, _ = self.getPoseAtHandle(self.target_handle, relative_handle, ignoreError, initialize)
		return pos

class rail(vrepObject):
	def __init__(self, clientID):
		super(rail, self).__init__(clientID)

		#Get handles
		self.rail_handle = self.getHandle('rail')

		self.dummy1_rail_handle = self.getHandle('rail_Dummy1')
		self.dummy2_rail_handle = self.getHandle('rail_Dummy2')
		self.dummy3_rail_handle = self.getHandle('rail_Dummy3')
		self.dummy4_rail_handle = self.getHandle('rail_Dummy4')
		self.dummy5_rail_handle = self.getHandle('rail_Dummy5')
		self.dummy6_rail_handle = self.getHandle('rail_Dummy6')
		self.dummy7_rail_handle = self.getHandle('rail_Dummy7')
		self.dummy8_rail_handle = self.getHandle('rail_Dummy8')

		#Achieved goals are three dummies on the bottom of the rail.
		#One is central, one is bottom and one is top (towards adrenal gland of kidney)
		self.rail_achieved_top = self.getHandle('rail_Achieved_t')
		self.rail_achieved_bottom = self.getHandle('rail_Achieved_b')
		self.rail_achieved_central = self.getHandle('rail_Achieved_goal')

		#Need respondable shape, a cuboid for the rail, for dynamics.
		#self.rail_res_handle = self.getHandle('rail_respondable')

	def setPose(self, pos, quat, grasp_site, relative_handle, ignoreError=False):
		b_T_d = self.posquat2Matrix(pos, quat)

		#Randomise grasping site:
		if grasp_site == 1:
			d_T_r = np.array([[1, 0, 0, 0], 
					  [0, 0, 1, 0.0325],
					  [0, -1, 0, -0.004],
					  [0, 0, 0, 1]])
			self.dummy_rail_handle = self.dummy1_rail_handle
		elif grasp_site == 2:
			d_T_r = np.array([[1, 0, 0, 0],
					  [0, 0, 1, 0.0225],
					  [0, -1, 0, -0.004],
					  [0, 0, 0, 1]])
			self.dummy_rail_handle = self.dummy2_rail_handle
		elif grasp_site == 3:
			d_T_r = np.array([[1, 0, 0, 0],
					  [0, 0, 1, 0.0125],
					  [0, -1, 0, -0.004],
					  [0, 0, 0, 1]])
			self.dummy_rail_handle = self.dummy3_rail_handle
		elif grasp_site == 4:
			d_T_r = np.array([[1, 0, 0, 0],
					  [0, 0, 1, 0.0025],
					  [0, -1, 0, -0.004],
					  [0, 0, 0, 1]])
			self.dummy_rail_handle = self.dummy4_rail_handle
		elif grasp_site == 5:
			d_T_r = np.array([[1, 0, 0, 0],
					  [0, 0, 1, -0.0075],
					  [0, -1, 0, -0.004],
					  [0, 0, 0, 1]])
			self.dummy_rail_handle = self.dummy5_rail_handle
		elif grasp_site == 6:
			d_T_r = np.array([[1, 0, 0, 0],
					  [0, 0, 1, -0.0175],
					  [0, -1, 0, -0.004],
				 	  [0, 0, 0, 1]])
			self.dummy_rail_handle = self.dummy6_rail_handle
		elif grasp_site == 7:
			d_T_r = np.array([[1, 0, 0, 0],
					  [0, 0, 1, -0.0275],
					  [0, -1, 0, -0.004],
					  [0, 0, 0, 1]])
			self.dummy_rail_handle = self.dummy7_rail_handle
		elif grasp_site == 8:
			d_T_r = np.array([[1, 0, 0, 0],
					  [0, 0, 1, -0.0375],
					  [0, -1, 0, -0.004],
					  [0, 0, 0, 1]])
			self.dummy_rail_handle = self.dummy8_rail_handle

		#Compute base to dummy position and quaternion
		pos, quat = self.matrix2posquat(np.dot(b_T_d, d_T_r))

		#For some reason I need to set the pose of respondable shape and rail even if it is parent.
		#self.setPoseAtHandle(self.rail_res_handle, relative_handle, pos, quat, ignoreError) 
		self.setPoseAtHandle(self.rail_handle, relative_handle, pos, quat, ignoreError) 

		#Return grasping site handle and its position.
		return self.dummy_rail_handle, pos

	def getPositionAchievedGoals(self, relative_handle, ignoreError=False, initialize=False):
		#Get position of dummies below the rail: central, top and bottom.
		pos_achieved_central, _ = self.getPoseAtHandle(self.rail_achieved_central, relative_handle, ignoreError, initialize)
		pos_achieved_top, _ = self.getPoseAtHandle(self.rail_achieved_top, relative_handle, ignoreError, initialize)
		pos_achieved_bottom, _ = self.getPoseAtHandle(self.rail_achieved_bottom, relative_handle, ignoreError, initialize)
		return pos_achieved_central, pos_achieved_top, pos_achieved_bottom

	def getPose(self, dummy_rail_handle, relative_handle, ignoreError=False, initialize=False):
		return self.getPoseAtHandle(dummy_rail_handle, relative_handle, ignoreError, initialize)

	def getVel(self, ignoreError=False, initialize=False):
		return self.getVelocityAtHandle(self.dummy_rail_handle, ignoreError, initialize)

	def removeGrasped(self, ignoreError=False):
		self.setParent(self.rail_handle, -1, True, ignoreError)
		#self.setParent(self.rail_res_handle, -1, True, ignoreError)

	def isGrasped(self, ignoreError=False, initialize=False):
		return not (-1 == self.getParent(self.rail_handle, ignoreError, initialize))
		#return not (-1 == self.getParent(self.rail_res_handle, ignoreError, initialize))


class targetK(vrepObject):
	'''
	@Info: class for target kidney. 
	'''
	def __init__(self, clientID):
		super(targetK, self).__init__(clientID)

		# Step 1: get the handles of the objects in the scene.
		# Parent
		self.k_res_handle = self.getHandle('Kidney_respondable')

		# Realistic Kidney
		self.k_handle = self.getHandle('Kidney')

		# Shape used for collision check
		self.convex = self.getHandle('Convex')

		# Surface dummies handles:
		# dh = dummy handle,
		# t = top (towards positive axis and adrenal gland), b = bottom
		self.k_dh_0t = self.getHandle('Kidney_Dummy_0t')
		self.k_dh_0b = self.getHandle('Kidney_Dummy_0b')
		self.k_dh_1t = self.getHandle('Kidney_Dummy_1t')
		self.k_dh_1b = self.getHandle('Kidney_Dummy_1b')
		self.k_dh_2t = self.getHandle('Kidney_Dummy_2t')
		self.k_dh_2b = self.getHandle('Kidney_Dummy_2b')
		self.k_dh_3t = self.getHandle('Kidney_Dummy_3t')
		self.k_dh_3b = self.getHandle('Kidney_Dummy_3b')
		self.k_dh_4t = self.getHandle('Kidney_Dummy_4t')
		self.k_dh_4b = self.getHandle('Kidney_Dummy_4b')
		#Dummies reached by the rail's central dummy below it
		self.k_dh_c0 = self.getHandle('Kidney_Dummy_c0')
		self.k_dh_c1 = self.getHandle('Kidney_Dummy_c1')
		self.k_dh_c2 = self.getHandle('Kidney_Dummy_c2')
		self.k_dh_c3 = self.getHandle('Kidney_Dummy_c3')
		self.k_dh_c4 = self.getHandle('Kidney_Dummy_c4')

		self.k_orientation_ctrl = self.getHandle('Kidney_orientation_ctrl')

		# Step 2: get the position of these objects,
		# otherwise the script can't tell where they are and prints "failed to get position and orientation".
		# -1 means relative to base frame.
		self.getPosition(-1, ignoreError = True, initialize = True)

	# This method sets the pose (position and quaternion) of the
	# cuboid shape, which then "sets" its children along with it.
	def setPose(self, pos, quat, relative_handle, ignoreError=False):
		self.setPoseAtHandle(self.k_res_handle, relative_handle, pos, quat, ignoreError)

	def getPosition(self, relative_handle, ignoreError=False, initialize = False):
		pos_res, _ = self.getPoseAtHandle(self.k_res_handle, relative_handle, ignoreError, initialize)
		pos_k, _ = self.getPoseAtHandle(self.k_handle, relative_handle, ignoreError, initialize)
		pos_convex, _ = self.getPoseAtHandle(self.convex, relative_handle, ignoreError, initialize)
		pos_0t, _ = self.getPoseAtHandle(self.k_dh_0t, relative_handle, ignoreError, initialize)
		pos_0b, _ = self.getPoseAtHandle(self.k_dh_0b, relative_handle, ignoreError, initialize)
		pos_1t, _ = self.getPoseAtHandle(self.k_dh_1t, relative_handle, ignoreError, initialize)
		pos_1b, _ = self.getPoseAtHandle(self.k_dh_1b, relative_handle, ignoreError, initialize)
		pos_2t, _ = self.getPoseAtHandle(self.k_dh_2t, relative_handle, ignoreError, initialize)
		pos_2b, _ = self.getPoseAtHandle(self.k_dh_2b, relative_handle, ignoreError, initialize)
		pos_3t, _ = self.getPoseAtHandle(self.k_dh_3t, relative_handle, ignoreError, initialize)
		pos_3b, _ = self.getPoseAtHandle(self.k_dh_3b, relative_handle, ignoreError, initialize)
		pos_4t, _ = self.getPoseAtHandle(self.k_dh_4t, relative_handle, ignoreError, initialize)
		pos_4b, _ = self.getPoseAtHandle(self.k_dh_4b, relative_handle, ignoreError, initialize)
		pos_c0, _ = self.getPoseAtHandle(self.k_dh_c0, relative_handle, ignoreError, initialize)
		pos_c1, _ = self.getPoseAtHandle(self.k_dh_c1, relative_handle, ignoreError, initialize)
		pos_c2, _ = self.getPoseAtHandle(self.k_dh_c2, relative_handle, ignoreError, initialize)
		pos_c3, _ = self.getPoseAtHandle(self.k_dh_c3, relative_handle, ignoreError, initialize)
		pos_c4, _ = self.getPoseAtHandle(self.k_dh_c4, relative_handle, ignoreError, initialize)
		pos_orientation_ctrl, _ = self.getPoseAtHandle(self.k_orientation_ctrl, relative_handle, ignoreError, initialize)

	# This method defines the random target used at goal.
	# This target is sampled off the 5 available ones. 
	def getPositionGoal(self, relative_handle, ignoreError=False, initialize=False):

		self.dummy_number = np.random.randint(0, 5)

		if self.dummy_number == 0:
			pos_c, _ = self.getPoseAtHandle(self.k_dh_c0, relative_handle, ignoreError, initialize)
			#print("Dummy pair PINK is goal.")
		elif self.dummy_number == 1:
			pos_c, _ = self.getPoseAtHandle(self.k_dh_c1, relative_handle, ignoreError, initialize)
			#print("Dummy pair GREEN is goal.")
		elif self.dummy_number == 2:
			pos_c, _ = self.getPoseAtHandle(self.k_dh_c2, relative_handle, ignoreError, initialize)
			#print("Dummy pair BLUE is goal.")
		elif self.dummy_number == 3:
			pos_c, _ = self.getPoseAtHandle(self.k_dh_c3, relative_handle, ignoreError, initialize)
			#print("Dummy pair YELLOW is goal.")
		else:
			pos_c, _ = self.getPoseAtHandle(self.k_dh_c4, relative_handle, ignoreError, initialize)
			#print("Dummy pair LILAC is goal.")

		return pos_c

   	# This method returns the top and bottom targets sampled by getPositionGoal.
	# They are front-facing dummies on opposite sides.
	# pos_t is the position of the top-side dummy,
	# pos_b the position of the bottom-side dummy.
	def getPositionGoalTopBottom(self, relative_handle, ignoreError=False, initialize=False):
		if self.dummy_number == 0:
			pos_t, _ = self.getPoseAtHandle(self.k_dh_0t, relative_handle, ignoreError, initialize)
			pos_b, _ = self.getPoseAtHandle(self.k_dh_0b, relative_handle, ignoreError, initialize)
		elif self.dummy_number == 1:
			pos_t, _ = self.getPoseAtHandle(self.k_dh_1t, relative_handle, ignoreError, initialize)
			pos_b, _ = self.getPoseAtHandle(self.k_dh_1b, relative_handle, ignoreError, initialize)
		elif self.dummy_number == 2:
			pos_t, _ = self.getPoseAtHandle(self.k_dh_2t, relative_handle, ignoreError, initialize)
			pos_b, _ = self.getPoseAtHandle(self.k_dh_2b, relative_handle, ignoreError, initialize)
		elif self.dummy_number == 3:
			pos_t, _ = self.getPoseAtHandle(self.k_dh_3t, relative_handle, ignoreError, initialize)
			pos_b, _ = self.getPoseAtHandle(self.k_dh_3b, relative_handle, ignoreError, initialize)
		else:
			pos_t, _ = self.getPoseAtHandle(self.k_dh_4t, relative_handle, ignoreError, initialize)
			pos_b, _ = self.getPoseAtHandle(self.k_dh_4b, relative_handle, ignoreError, initialize)

		return pos_t, pos_b

	#This method returns the orientation of Kidney_orientation_ctrl. The dummy
	#used in _align_to_target() in PsmEnv_Position.py to compute the orientation error
	#between the EE and the target. 
	def getOrientationGoals(self, relative_handle, ignoreError=False, initialize=False):
		pos, quat = self.getPoseAtHandle(self.k_orientation_ctrl, relative_handle, ignoreError, initialize)
		return quat

class collisionCheck(vrepObject):
	'''
	@Info:	
	Method for checking collision of collision objects in the scene.
	@Returns: the collision states of the collision objects in the scene.
	'''
	def __init__(self, clientID, psm_number):
		super(collisionCheck, self).__init__(clientID)

		#Collision objects of Rail. Check against the kidney, cuboid and convex shell.
		self.collision_Kidney_Rail = self.getCollisionHandle('Collision_Kidney_Rail')
		self.collision_Cuboid_Rail = self.getCollisionHandle('Collision_Cuboid_Rail')
		self.collision_Convex_Rail = self.getCollisionHandle('Collision_Convex_Rail')

		#Collision objects of Robot. Check against convex all the possible colliding parts.
		#These are, the black cylinder, the TT's body and tips.
		self.collision_Convex_Cylinder = self.getCollisionHandle('Collision_Convex_Cylinder')
		self.collision_Convex_TT_body = self.getCollisionHandle('Collision_Convex_TT_body')
		self.collision_Convex_TT_sx = self.getCollisionHandle('Collision_Convex_TT_sx')
		self.collision_Convex_TT_dx = self.getCollisionHandle('Collision_Convex_TT_dx')
        
		#Init
		super(collisionCheck, self).checkCollision(self.collision_Kidney_Rail, ignoreError=True, initialize=True)
		super(collisionCheck, self).checkCollision(self.collision_Cuboid_Rail, ignoreError=True, initialize=True)
		super(collisionCheck, self).checkCollision(self.collision_Convex_Rail, ignoreError=True, initialize=True)
		super(collisionCheck, self).checkCollision(self.collision_Convex_Cylinder, ignoreError=True, initialize=True)
		super(collisionCheck, self).checkCollision(self.collision_Convex_TT_body, ignoreError=True, initialize=True)
		super(collisionCheck, self).checkCollision(self.collision_Convex_TT_sx, ignoreError=True, initialize=True)
		super(collisionCheck, self).checkCollision(self.collision_Convex_TT_dx, ignoreError=True, initialize=True)

	# Any checkCollision call returns True if collision and False if not collision
	def KidneyCollision(self, ignoreError=False):
		#c_r = collision result
		c_r1 = super(collisionCheck,self).checkCollision(self.collision_Kidney_Rail, ignoreError)
		c_r2 = super(collisionCheck,self).checkCollision(self.collision_Cuboid_Rail, ignoreError)
		c_r3 = super(collisionCheck,self).checkCollision(self.collision_Convex_Rail, ignoreError)
		c_r4 = super(collisionCheck,self).checkCollision(self.collision_Convex_Cylinder, ignoreError)
		c_r5 = super(collisionCheck,self).checkCollision(self.collision_Convex_TT_body, ignoreError)
		c_r6 = super(collisionCheck,self).checkCollision(self.collision_Convex_TT_sx, ignoreError)
		c_r7 = super(collisionCheck,self).checkCollision(self.collision_Convex_TT_dx, ignoreError)
		return c_r1, c_r2, c_r3, c_r4, c_r5, c_r6, c_r7
