import gym
import logging
import math
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
import pybullet as p
import pkg_resources
import xml.etree.ElementTree as etxml
import pybullet_data
from collections import deque


class PathFinderEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
               init_xyzs=[0,0,0.0],
               final_xyzs=[0,0,0.0],
               init_RPYs=[0,0,0],
               final_yaw=0,
               gui=False,
               sim_freq=120):
        self.mem_count= 64
    
        # Hyperparameter definition 
        self.x_min = int(-1)
        self.x_max = int(1)
        self.y_min = int(-1)
        self.y_max = int(1)
        self.z_min = int(-1)
        self.z_max = int(1)

        self.init_xyzs = np.array(init_xyzs)
        self.final_xyzs = np.array(final_xyzs)
        self.init_RPYs = np.array(init_RPYs)
        self.final_RPYs = np.array([0,0,final_yaw])

        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        self.xyz = self.init_xyzs
        self.RPY = self.init_RPYs
        self.quat = p.getQuaternionFromEuler(init_RPYs)
        self.final_quat = p.getQuaternionFromEuler(self.final_RPYs)
        quat_scaled = (np.array(self.quat)+1)*128
        self.curr_state = np.array([*quat_scaled])
        self.state_memory = deque(maxlen=self.mem_count)
        for i in range(self.mem_count):
            self.state_memory.append(self.curr_state)

        # Drone data
        self.hover_rpm = 14500
        self.max_rpm = self.hover_rpm*2
        self.KF, self.KM = [0,0]
        self._parseURDFParameters()
        self.DRAG_COEFF = 9.1785e-7
        
        # Sim data
        self.SIM_FREQ = sim_freq
        self.TIMESTEP = 1./self.SIM_FREQ
        self.G = 9.8

        # ???
        self.episode_over = False
        self.current_episode = 0
        
        # Here, low is the lower limit of observation range, and high is the higher limit.
        self.observation_space = spaces.MultiDiscrete([64, 64, 64, 64]*self.mem_count)
        
        # Action space
        self.action_space = spaces.MultiDiscrete([64]*4)

        self.physicsSetup()

    def physicsSetup(self):
        p.setGravity(0, 0, -self.G, physicsClientId=self.client)
        p.setRealTimeSimulation(0, physicsClientId=self.client)
        p.setTimeStep(self.TIMESTEP, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        self.drone = p.loadURDF(pkg_resources.resource_filename('path_finder', 'assets/drone.urdf'),
                            self.init_xyzs,
                            p.getQuaternionFromEuler(self.init_RPYs),
                            flags = p.URDF_USE_INERTIA_FROM_FILE,
                            physicsClientId=self.client
                          )
        # self.plane_id = p.loadURDF("plane.urdf", [0,0,-1], physicsClientId=self.client)

    def step(self, action):
        
        """
            The agent (drone) takes a step (flies somewhere) in the environment.
            Parameters
            ----------
     action : (int,int) - the coordinates, (int) - the terrain gradient
            Returns: (int) - terrain angle (observation), (float32) reward, (bool) episode_over, (int,int) - coords
            -------
            ob, reward, episode_over, info : tuple
                ob (object) :
                    an environment-specific object representing your observation of
                    the environment.
                reward (float) :
                    amount of reward achieved by the previous action. The scale
                    varies between environments, but the goal is always to increase
                    your total reward. (This reward per step is normalised to 1.)
                episode_over (bool) :
                    whether it's time to reset the environment again. Most (but not
                    all) tasks are divided up into well-defined episodes, and done
                    being True indicates the episode has terminated. (For example,
                    perhaps the pole tipped too far, or you lost your last life.)
                info (dict) :
                     diagnostic information useful for debugging. It can sometimes
                     be useful for learning (for example, it might contain the raw
                     probabilities behind the environment's last state change).
                     However, official evaluations of your agent are not allowed to
                     use this for learning.
            """
        
        if self.episode_over:
            raise RuntimeError("Episode is done. You're running step() despite this fact. Or reset the env by calling reset().") #end execution, and finish run

        action_rpm = self.max_rpm*action/256
        self._physics(action_rpm)
        p.stepSimulation(physicsClientId=self.client)

        self._updateInput()
        self.curr_state = self._normalize_state()
        self.curr_state += 1
        self.curr_state *= 128
        self.state_memory.append(self.curr_state)

        # Return the reward for action taken given state. Save action to action memory buffer.
        reward = self._get_reward()

        # Take a step, and observe environment.
        self.current_timestep += 1

        # if (self.xyz[0] > self.x_max) or (self.xyz[0] < self.x_min) \
        # or (self.xyz[1] > self.y_max) or (self.xyz[1] < self.y_min) \
        # or (self.xyz[2] > self.z_max) or (self.xyz[2] < self.z_min):
        #     self.episode_over = True

        ret_state = np.concatenate(self.state_memory, axis=0)
        return ret_state, reward, self.episode_over, {}

    def _physics(self,
                 rpm,
                 ):
        """Base PyBullet physics implementation.

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        """
        # Propeller forces
        forces = np.array(rpm**2)*self.KF
        torques = np.array(rpm**2)*self.KM
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])
        for i in range(4):
            p.applyExternalForce(self.drone,
                                 i,
                                 forceObj=[0, 0, forces[i]],
                                 posObj=[0, 0, 0],
                                 flags=p.LINK_FRAME,
                                 physicsClientId=self.client
                                 )
        p.applyExternalTorque(self.drone,
                              4,
                              torqueObj=[0, 0, z_torque],
                              flags=p.LINK_FRAME,
                              physicsClientId=self.client
                              )

        # Drag
        base_rot = np.array(p.getMatrixFromQuaternion(self.quat)).reshape(3, 3)
        #### Simple draft model applied to the base/center of mass #
        drag_factors = -1 * self.DRAG_COEFF * np.sum(np.array(2*np.pi*rpm/60))
        drag = np.dot(base_rot, drag_factors*np.array(self.Lv))
        p.applyExternalForce(self.drone,
                             4,
                             forceObj=drag,
                             posObj=[0, 0, 0],
                             flags=p.LINK_FRAME,
                             physicsClientId=self.client
                             )

    def _get_info(self):
        return self.xyz

    def _updateInput(self):
        self.xyz, self.quat = p.getBasePositionAndOrientation(self.drone, physicsClientId=self.client)
        self.RPY = p.getEulerFromQuaternion(self.quat)

    def _normalize_state(self):
        state = np.hstack([self.quat]).reshape(4,)
        return state
  
    def _get_reward(self):

        xy_reward = 0.3*np.tanh(1-(abs(self.xyz[:2] - self.final_xyzs[:2])).sum())
        z_reward = np.tanh(1-(abs(self.xyz[-1] - self.final_xyzs[-1])).sum())
        position_reward = xy_reward + z_reward
        
        attitude_reward = 0.1*np.tanh(1-(abs(np.array(self.quat) - np.array(self.final_quat))).sum())

        return position_reward + attitude_reward


    def reset(self):
        # reset should always run at the end of an episode and before the first run.
        self.current_timestep = 0
        self.episode_over = False
        self.xyz = self.init_xyzs
        self.RPY = self.init_RPYs
        self.quat = p.getQuaternionFromEuler(self.init_RPYs)
        quat_scaled = (np.array(self.quat)+1)*128
        self.curr_state = quat_scaled
        self.state_memory = deque(maxlen=self.mem_count)
        for i in range(self.mem_count):
            self.state_memory.append(self.curr_state)
        p.resetBasePositionAndOrientation(self.drone,
                                          self.xyz,
                                          self.quat,
                                          physicsClientId=self.client
                                          )
        p.setRealTimeSimulation(0, physicsClientId=self.client)
        return np.concatenate([self.curr_state]*self.mem_count, axis=0)
    
    def render(self, mode='human'):
        return 0
    def _render(self, mode='human', close=False):
        return 0
    def close(self):
        p.disconnect(physicsClientId=self.client)
        return 0

    def _parseURDFParameters(self):
        """Loads parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.

        """
        URDF_TREE = etxml.parse(pkg_resources.resource_filename('path_finder', 'assets/drone.urdf')).getroot()
        self.KF = float(URDF_TREE[0].attrib['kf'])
        self.KM = float(URDF_TREE[0].attrib['km'])
