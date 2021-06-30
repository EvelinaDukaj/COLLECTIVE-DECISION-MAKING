from __future__ import division, print_function, absolute_import
import copy
import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import datetime
from cdm_project import approachActivation


from gym import spaces
from gym.utils import seeding
from highwayEnv.Vars_and_Methods import methods, variables
from highwayEnv.environment.observation import observation_factory
from highwayEnv.vehicles.control import AgentVehicle, simpleVehicle
from gym.envs.registration import register
from highwayEnv.road.lane import LineType, StraightLane, CircularLane, SineLane, AbstractLane
from highwayEnv.road.road import Road, RoadNetwork
from highwayEnv.analytics import build_boxplot, build_scatterplot, calculate_statistics
from highwayEnv.environment.archive import Archive



class RoundaboutEnv(gym.Env):
    """
        A generic environment for a vehicle driving on a roundabout.
        The action space is fixed, but the observation space and reward function must be defined in the
        environment implementations.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    ACTIONS = variables.ACTIONS
    """
        A mapping of action indexes to action labels
    """
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}
    """
        A mapping of action labels to action indexes
    """

    SIMULATION_FREQUENCY = variables.SIMULATION_FREQUENCY
    """
        The frequency at which the system dynamics are simulated [Hz]
    """

    PERCEPTION_DISTANCE = 5.0 * AgentVehicle.SPEED_MAX
    """
        The maximum distance of any vehicle present in the observation [m]
    """
    TIME_DIFFERENCE = variables.time_difference
    """ 
        Inter arrival time of ego vehicles 
    """
    
    # For plots
    plotCounter = 0

    DEFAULT_CONFIG = {
        "observation": {
            "type": "Kinematics"
        },
        "policy_frequency": 1,  # [Hz]
        "other_vehicles_type": "highwayEnv.vehicles.control.simpleVehicle",
        "incoming_vehicle_destination": None,
        "screen_width": 800,
        "screen_height": 750,
        "centering_position": [0.9, 0.1]
    }
    size_prev = 0

    def __init__(self, config=None):

        """------------------- Variables to configure for the Paralelization ------------------- """
        self.sigma = 0
        self.risk_tol = 0
        self.threshold_tol = 0
        self.hv_tol = 0
        self.total_attempts = 0

        # Test Metric
        self.cumulatedCrashes = 0
        self.cumulatedTime = 0

        # Configuration
        self.config = config
        if not self.config:
            self.config = self.DEFAULT_CONFIG.copy()

        # Seeding
        self.np_random = None

        # Scene
        self.road = None
        self.other_vehicles = []
        self.ego_vehicles = []

        # Spaces
        self.observation = None
        self.define_spaces()

        # Running
        self.done = False

        # Rendering
        self.viewer = None
        self.automatic_rendering_callback = None
        self.should_update_rendering = True
        self.rendering_mode = 'human'
        self.offscreen = self.config.get("offscreen_rendering", True)
        
        self.enable_auto_render = True

        self.steps = 0

        self.selected_seed = 0
        self.distances_to_solutions = []
        self.archive = Archive()


    

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def configure(self, config):
        if config:
            self.config.update(config)

    def define_spaces(self):
        self.action_space = spaces.Discrete(len(self.ACTIONS))

        if "observation" not in self.config:
            raise ValueError("The observation configuration must be defined")
        self.observation = observation_factory(self, self.config["observation"])
        self.observation_space = self.observation.space()

    def _is_terminal(self):
        """
            Check whether the current state is a terminal state
        :return:is the state terminal
        """

        if (self.total_attempts >= variables.test_terminal):
            return True, calculate_statistics(self.archive.get_archive(), self.cumulatedCrashes, self.cumulatedTime, self.risk_tol,
                    self.threshold_tol, self.hv_tol, self.sigma, self.total_attempts, self.steps, self.selected_seed, self.distances_to_solutions)
        else:
            return False, pd.DataFrame()


    def reset(self):
        """
            Reset the environment to it's initial configuration
        :return: the observation of the reset state
        """
        self.other_vehicles = []
        self.ego_vehicles = []
        self._make_road()
        self._make_other_vehicles()
        self.steps = 0
        self.done = False
        self.define_spaces()
        return self.observation.observe()

    def step(self, action):
        """
            Perform an action and step the environment dynamics.

            The action is executed by the ego-vehicle, and all other vehicles on the road performs their default
            behaviour for several simulation timesteps until the next decision making step.
        :param int action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminal, info)
        """
        self._make_ego_vehicles()
        if self.road is None or self.ego_vehicles is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")
        
        self.steps += 1
        for agent in self.ego_vehicles:
            if agent.lane_index ==("nxr", "sen" ,0) and not agent.route:
                agent.plan_route_to("nes")
            elif agent.lane_index == ("nes", "ne", 0) and not agent.route:
                agent.plan_route_to("nxr")
            
            #after leaving roundabout, clear our solution selected
            if agent.lane_index in [("sxn","ner", 0), ("nx","nxs", 0)] and agent.stats_updated==False:
                self.cumulatedCrashes += agent.crashed
                self.cumulatedTime += agent.waiting_time
                agent.stats_updated = True
                self.total_attempts += 1
                agent.stopped = False
            
            #on entrance lane to roundabout
            if agent.lane_index in [("nxr", "sen", 0), ("nes", "ne", 0)]:
                agent.roundabout_entrance_test(self)     
            else: 
                agent.act(self.ACTIONS[action])

        #To save information for statistics:
        self.road.act()
        self.road.step(1 / self.SIMULATION_FREQUENCY, self.ego_vehicles, self.other_vehicles)
        methods._automatic_rendering(self)
        self.enable_auto_render = False
        return self._is_terminal()

    def close(self):
        """
            Close the environment.

            Will close the environment viewer if it exists.
        """
        self.done = True
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None
        plt.close('all')
        #return self.done
        #self.close()
        

    def get_available_actions(self):
        """
            Get the list of currently available actions.

            Lane changes are not available on the boundary of the road, and velocity changes are not available at
            maximal or minimal velocity.

        :return: the list of available actions
        """
        actions = [self.ACTIONS_INDEXES['IDLE']]
        if self.vehicle.velocity_index < self.vehicle.SPEED_COUNT - 1:
            actions.append(self.ACTIONS_INDEXES['FASTER'])
        if self.vehicle.velocity_index > 0:
            actions.append(self.ACTIONS_INDEXES['SLOWER'])
        actions.append(self.ACTIONS_INDEXES['STOP'])
        return actions

        """ 
            vehicle collections:
            all cars including ego-vehicles: RoundaboutEnv.road.vehicles
            ego vehicles: RoundaboutEnv.vehicle
        """
    def _make_road(self):
        # Circle lanes: (s)outh/(e)ast/(n)orth/(w)est (e)ntry/e(x)it.
        center = variables.center_south  # [m]
        centerNorth = variables.center_north  # [m]
        radius = variables.raradius  # [m]
        alpha = 20  # [deg]

        net = RoadNetwork()
        radii = radius+4
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [c, c]
    
        """  
            Creation of south Roundabout
        """
        net.add_lane("se", "ex", CircularLane(center, radii, rad(90 - alpha), rad(alpha), clockwise=False, line_types=line))
        net.add_lane("ex", "ee", CircularLane(center, radii, rad(alpha), rad(-alpha), clockwise=False, line_types=line))
        net.add_lane("ee", "nx", CircularLane(center, radii, rad(-alpha), rad(-90 + alpha), clockwise=False, line_types=line))
        net.add_lane("nx", "ne", CircularLane(center, radii, rad(-90 + alpha), rad(-90 - alpha), clockwise=False, line_types=line))
        net.add_lane("ne", "wx", CircularLane(center, radii, rad(-90 - alpha), rad(-180 + alpha), clockwise=False, line_types=line))
        net.add_lane("wx", "we", CircularLane(center, radii, rad(-180 + alpha), rad(-180 - alpha), clockwise=False, line_types=line))
        net.add_lane("we", "sx", CircularLane(center, radii, rad(180 - alpha), rad(90 + alpha), clockwise=False, line_types=line))
        net.add_lane("sx", "se", CircularLane(center, radii, rad(90 + alpha), rad(90 - alpha), clockwise=False, line_types=line))

        """ 
            Creation of North Roundabout
        """

        # Circle lanes: (s)outh/(e)ast/(n)orth/(w)est + (e)ntry/e(x)it + n.
        net.add_lane("sen", "exn", CircularLane(centerNorth, radii, rad(90 - alpha), rad(alpha), clockwise=False, line_types=line))
        net.add_lane("exn", "een", CircularLane(centerNorth, radii, rad(alpha), rad(-alpha), clockwise=False, line_types=line))
        net.add_lane("een", "nxn", CircularLane(centerNorth, radii, rad(-alpha), rad(-90 + alpha), clockwise=False, line_types=line))
        net.add_lane("nxn", "nen", CircularLane(centerNorth, radii, rad(-90 + alpha), rad(-90 - alpha), clockwise=False, line_types=line))
        net.add_lane("nen", "wxn", CircularLane(centerNorth, radii, rad(-90 - alpha), rad(-180 + alpha), clockwise=False, line_types=line))
        net.add_lane("wxn", "wen", CircularLane(centerNorth, radii, rad(-180 + alpha), rad(-180 - alpha), clockwise=False, line_types=line))
        net.add_lane("wen", "sxn", CircularLane(centerNorth, radii, rad(180 - alpha), rad(90 + alpha), clockwise=False, line_types=line))
        net.add_lane("sxn", "sen", CircularLane(centerNorth, radii, rad(90 + alpha), rad(90 - alpha), clockwise=False, line_types=line))


        # Access lanes: (r)oad/(s)ine
        access = 150  # [m]
        dev = 120  # [m]
        a = 5  # [m]
        delta_st = 0.20 * dev  # [m]

        delta_en = dev - delta_st
        w = 2 * np.pi / dev

        #Lanes to create the start of the ego_cars
        net.add_lane("start", "east", StraightLane([access, -access/2], [70, -access/2], width=60, line_types=line, speed_limit=50))
        net.add_lane("east", "easte", StraightLane([70, -access/2], [30,-access/2], line_types=line))
        net.add_lane("easte", "mid", SineLane([30, -access/2-a], [4, -access/2-a], a, w, -np.pi / 2, line_types=line))
        net.add_lane("none", "none", StraightLane([70, -access/2-2], [70,-105], width=0, line_types=line))
        net.add_lane("none", "none", StraightLane([70, -73], [70,-45], width=0, line_types=line))

        #Lines to go to the north roundabout
        net.add_lane("ner", "nes", StraightLane([-2, -access], [-2, -dev / 2], line_types=[s, c]))
        net.add_lane("nes", "ne", SineLane([-2 - a, -dev / 2], [-2 - a, -dev / 2 + delta_st], a, w, -np.pi / 2, line_types=line))
        
        #Lines to go from south to north, mid split the straightline so the cars can enter from the pool
        net.add_lane("nx", "nxs", SineLane([2 + a, dev / 2 - delta_en], [2 + a, -dev / 2], a, w, -np.pi / 2 + w * delta_en, line_types=line))
        net.add_lane("nxs", "mid", StraightLane([2, -dev / 2], [2, -access/2], line_types=[n, c]))
        net.add_lane("mid", "nxr", StraightLane([2, -access/2], [2, -access], line_types=[n, c]))

        #Lines for enter and exit the north round about
        net.add_lane("sxn", "ner", SineLane([-2-a, -access-delta_st], [-2-a, -access], a, w, -np.pi/2+w*delta_en, line_types=line))
        net.add_lane("nxr", "sen", SineLane([2+a, -access], [2+a, -access-delta_st], a, w, -np.pi/2, line_types=line))       
        #net.add_lane("sem", "sen", SineLane([2+a, -access-delta_st+5], [2+a, -access-delta_st], a, w, -np.pi/2+w*delta_en, line_types=line)) 
        road = Road(network=net, np_random=self.np_random)
        self.road = road

    def _make_other_vehicles(self):
        """
            Populate the roundabouts with dummy vehicles (Blue vehicles)
        """
        position_deviation = 2
        
        # Other vehicles

        other_vehicles_type = methods.class_from_path(self.config["other_vehicles_type"])

        #Roundabout south
        for i in range(1, variables.num_other_cars_south+1):
            if i == 1: 
                vehicle = other_vehicles_type.make_on_lane(self.road,
                                                       ("ne", "wx", 0),
                                                       longitudinal=20 * i + self.np_random.randn() * position_deviation,
                                                       velocity=variables.MAX_VELOCITY)
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
                self.other_vehicles.append(vehicle)
                #vehicle.setDistanceWanted(30)
                
            else:
                vehicle = other_vehicles_type.make_on_lane(self.road,
                                                       ("ne", "wx", 0),
                                                       longitudinal=20*i + self.np_random.randn() * position_deviation,
                                                       velocity=variables.MAX_VELOCITY)
                vehicle.randomize_behavior()
                vehicle.setDistanceWanted(np.random.random_sample() + 0.1)
                self.road.vehicles.append(vehicle)
                self.other_vehicles.append(vehicle)
                #vehicle.setDistanceWanted(0)

        #Roundabout north
        for i in range(1, variables.num_other_cars_north+1):
            if i == 1 : 
                vehicle = other_vehicles_type.make_on_lane(self.road,
                                                       ("wen", "sxn", 0),
                                                       longitudinal=20 * i + self.np_random.randn() * position_deviation,
                                                       velocity=variables.MAX_VELOCITY)
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
                self.other_vehicles.append(vehicle)
                
                
            else:
                vehicle = other_vehicles_type.make_on_lane(self.road,
                                                       ("wen", "sxn", 0),
                                                       longitudinal=20 * i + self.np_random.randn() * position_deviation,
                                                       velocity=variables.MAX_VELOCITY)
                vehicle.randomize_behavior()
                vehicle.setDistanceWanted(29 * np.random.random_sample() + 1) #self.np_random.randn() * 29
                self.road.vehicles.append(vehicle)
                self.other_vehicles.append(vehicle)
                
    
    def _make_ego_vehicles(self):

        """ 
            Ego-vehicle
            Creates Ego-vehicle on entrance lane to roundabout 
        """
        ego_lane = self.road.network.get_lane(variables.START_POS)

        # ego_lane.position is where the car will appear, and heading at is the direction

        if (self.steps%self.TIME_DIFFERENCE == 0 and len(self.ego_vehicles) < variables.num_ego_vehicles):
            longitud = random.randint(0, 50)
            lateral = random.randint(-35, 35)
            ego_vehicle = AgentVehicle(self.road,
                                     ego_lane.position(longitud,lateral), 
                                     velocity=variables.MAX_VELOCITY,
                                     heading=ego_lane.heading_at(3), 
                                     id=str(len(self.ego_vehicles))).plan_route_to("nxr")
            self.road.vehicles.append(ego_vehicle)
            self.ego_vehicles.append(ego_vehicle)

            
    #Define the parameter settings for the run
    def set_configuration(self, risk_tol,threshold_tol,hv_tol,sigma,seed):
        self.risk_tol = risk_tol
        self.threshold_tol = threshold_tol
        self.hv_tol = hv_tol
        self.sigma = sigma
        np.random.seed(seed)
        random.seed(seed)
        self.seed(seed)
        self.reset()
        self.selected_seed = seed

        # Archive for thresholds
        self.archive.set_archive(self.risk_tol,self.threshold_tol,self.hv_tol,self.sigma)

        if approachActivation()[2]== True:
            self.archive.set_archive(self.risk_tol,self.threshold_tol,self.hv_tol,self.sigma)


def rad(deg):
        return deg * np.pi / 180



register(
        id='roundaboutTest-v1',
        entry_point='highwayEnv.Tests:RoundaboutEnv',
    )
