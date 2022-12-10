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


class PathFinderEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
               init_xyzs=np.array([0,0,0.5]),
               final_xyzs=np.array([0,0,0.5]),
               init_RPYs=np.array([0,0,0]),
               final_yaw=0,
               gui=False,
               sim_freq=120):
    
        # Hyperparameter definition 
        self.x_min = int(-3)
        self.x_max = int(3)
        self.y_min = int(-3)
        self.y_max = int(3)
        self.z_min = int(-3)
        self.z_max = int(3) #meter

        self.init_xyzs = init_xyzs
        self.final_xyzs = init_xyzs
        self.init_RPYs = init_RPYs

        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        self.drone = p.loadURDF(pkg_resources.resource_filename('path_finder', 'assets/drone.urdf'),
                                self.init_xyzs,
                                p.getQuaternionFromEuler(self.init_RPYs),
                                flags = p.URDF_USE_INERTIA_FROM_FILE,
                                physicsClientId=self.client
                              )

        self.xyz = init_xyzs
        self.RPY = init_RPYs
        self.state = self._get_state()
        self.action_memory = []

        # Drone data
        self.max_rpm = 20000
        self.KF, self.KM = [0,0]
        self._parseURDFParameters()
        
        # Sim data
        self.SIM_FREQ = sim_freq
        self.TIMESTEP = 1./self.SIM_FREQ
        self.G = 9.8

        # ???
        self.episode_over = False
        self.current_episode = 0
        
        # Here, low is the lower limit of observation range, and high is the higher limit.
        low_ob = np.array([-1,-1,-1, -1,-1,-1]) # x y z R P Y * 6 windows
        high_ob = np.array([1,1,1, 1,1,1])
        self.observation_space = spaces.Box(low_ob, high_ob,
                                            shape=(6,),
                                            dtype=np.float16)
        
        # Action space
        low_action = np.array([0,0,0,0], dtype=np.float16) # RPM control
        high_action = np.array([1,1,1,1], dtype=np.float16) # RPM control
        self.action_space = spaces.Box(low_action, high_action,
                                            shape=(4,),
                                            dtype=np.float16)

        self.physicsSetup()
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)

    def physicsSetup(self):
        p.setGravity(0, 0, -self.G, physicsClientId=self.client)
        p.setRealTimeSimulation(0, physicsClientId=self.client)
        p.setTimeStep(self.TIMESTEP, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
    
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

        # process action
        action = np.squeeze(action)
        action += 1
        action /= 2
        self.last_action = action

        self._physics(action*self.max_rpm)
        p.stepSimulation(physicsClientId=self.client)
        self._updateInput()
        self.state = self._get_state()

        # Return the reward for action taken given state. Save action to action memory buffer.
        # self.action_memory.append(action)
        reward = self._get_reward()

        # Take a step, and observe environment.
        self.current_timestep += 1

        # if (self.xyz[0] > self.x_max) or (self.xyz[0] < self.x_min) \
        # or (self.xyz[1] > self.y_max) or (self.xyz[1] < self.y_min) \
        # or (self.xyz[2] > self.z_max) or (self.xyz[2] < self.z_min):
        #     # self.episode_over = True
        #     reward = -1000
        
        return self.state, reward, self.episode_over, {}

    def _physics(self,
                 rpm,
                 ):
        """Base PyBullet physics implementation.

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
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

    def _get_info(self):
        return self.xyz

    def _updateInput(self):
        self.xyz, self.quat = p.getBasePositionAndOrientation(self.drone, physicsClientId=self.client)
        self.RPY = p.getEulerFromQuaternion(self.quat)

    def _get_state(self):

        # Clip stuff
        MAX_PITCH_ROLL = np.pi
        clipped_RP = np.clip(self.RPY[:2], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        norm_RP = clipped_RP / MAX_PITCH_ROLL
        norm_Y = self.RPY[2] / MAX_PITCH_ROLL

        MAX_XY = 3
        MAX_Z = 3
        clipped_xy = np.clip(self.xyz[:2], -MAX_XY, MAX_XY)
        clipped_z = np.clip(self.xyz[2], 0, MAX_Z)
        normalized_pos_xy = clipped_xy / MAX_XY
        normalized_pos_z = clipped_z / MAX_Z

        state = np.hstack([normalized_pos_xy, normalized_pos_z, norm_RP, norm_Y]).reshape(6,)
        return state
  
    def _get_reward(self):
        x, y, z, = self.xyz[0], self.xyz[1], self.xyz[2]
        fx, fy, fz = self.final_xyzs[0], self.final_xyzs[1], self.final_xyzs[2]

        # reward = 0
        # if abs(x - fx) < 0.05:
        #     reward+=1
        # else:
        #     reward-=1
        # if abs(y - fy) < 0.05:
        #     reward+=0.1
        # if abs(z - fz) < 0.05:
        #     reward+=0.1
        # return reward

        # position_reward = -np.sqrt(((self.xyz[:3]-self.final_xyzs)**2).sum())
        # euler_reward = -np.sqrt(((self.RPY[:2]-np.array([0,0]))**2).sum())
        # action_reward = self.last_action.sum()

        # return position_reward# + euler_reward + action_reward
    
        return np.tanh(1-0.3*(abs(self.xyz[:3] - self.final_xyzs)).sum())


    def reset(self):
        # reset should always run at the end of an episode and before the first run.
        self.current_timestep = 0
        self.action_memory = []
        self.episode_over = False
        self.state[:3] = self.init_xyzs
        self.state[3:] = self.init_RPYs
        
        self.physicsSetup()
        return self.state
    
    def render(self, mode='human'):
        return 0
    def _render(self, mode='human', close=False):
        return 0
    def close(self):
        return 0

    def _parseURDFParameters(self):
        """Loads parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.

        """
        URDF_TREE = etxml.parse(pkg_resources.resource_filename('path_finder', 'assets/drone.urdf')).getroot()
        self.KF = float(URDF_TREE[0].attrib['kf'])
        self.KM = float(URDF_TREE[0].attrib['km'])
