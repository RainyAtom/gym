import gym.spaces
from gym import spaces
from gym.utils import seeding
from gym.envs.rover_domain.simulators.rover_domain_simulator import RoverDomainSim
from gym.envs.rover_domain import rendering
import yaml
import sys

class RoverDomain(gym.Env):
    metadata = {
        'render.modes': ['human', 'rbg_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        # Read and store parameters from configuration file.
        if len(sys.argv) is 1:
            config_f = "config.yml"
        else:
            config_f = sys.argv[1]
        with open(config_f, 'r') as f:
            config = yaml.load(f)

        # Initialize the rover domain.
        self.domain = RoverDomainSim(
            config["Seed"],
            config["Initial POI Locations"],
            config["Initial Agent Positions"],
            config["Number of Agents"],
            config["Number of POIs"],
            config["World Width"],
            config["World Length"])

        self.poi_loc = self.domain.get_jointstate()['pois']
        self.agent_loc = self.domain.get_jointstate()['agents']

        self.num_pois = config["Number of POIs"]
        self.num_agents = config["Number of Agents"]

        self.world_width = config["World Width"]
        self.world_length = config["World Length"]
        self.observation_rad = config["Observation Radius"]

        # Dict to store agent locations
        self.path = {}
        # Number of steps to execute before tracking paths
        self.step_path_flag = (config["Epochs"] * config["Steps"]) - config["Steps"]
        self.current_step = 0

        self.viewer = None

        # self.action_space = spaces.Discrete(self.world_width)
        # self.observation_space = self.domain.get_jointstate()
        self.seed()
        self.reset()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, actions):
        # Pass actions to domain to update
        self.domain.apply_actions(actions)
        # Update the joint state
        joint_state = self.domain.get_jointstate()

        # Increment step executed
        self.current_step = self.current_step + 1
        # Store agent locations
        if self.current_step == (self.step_path_flag + 1):
            for a in range(self.num_agents):
                self.path["agent" + str(a)] = [tuple(self.agent_loc['agent_' + str(a)]['loc'])]
        elif self.current_step > (self.step_path_flag + 1):
            for i in range(self.num_agents):
                self.path["agent" + str(i)].append(tuple(joint_state['agents']['agent_' + str(i)]['loc']))

        done = True # serves no purpose atm
        reward = 0  # serves no purpose atm

        return joint_state, reward, done, {}


    def reset(self):
        return self.domain.get_jointstate()


    def render(self, mode='human'):

        screen_width = 550
        screen_height = 550
        # Number subtracted to create border around domain rendering,
        # half of this number needs to be added to the locations of all objects, to center rendering
        scale = (screen_width - 50) / self.world_width
        agent_side = 10

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.agent_trans = []

            # Render agents
            color = 0
            for i in range(self.num_agents):
                l, r, t, b = -agent_side/2, agent_side/2, agent_side/2, -agent_side/2   # left, right, top, bottom
                agent = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
                # Change color for each agent
                color = color + (1 / self.num_agents)
                agent.set_color(0, color, 0)    #rgb
                agenttrans = rendering.Transform()
                agent.add_attr(agenttrans)
                self.agent_trans.append(agenttrans)
                self.viewer.add_geom(agent)

            # Render POIs and their observation radius
            color = 0
            for i in range(self.num_pois):
                # Customize the radius size of a POI according to its value
                poi_rad = 5 + (self.poi_loc['poi_' + str(i)]['value'] / 2)
                poi = rendering.make_circle(poi_rad)
                observation_rad = rendering.make_circle(self.observation_rad*scale, 30, False)
                # Change color for each POI
                color = color + (1 / self.num_pois)
                poi.set_color(color, 0, 0)  #rgb
                observation_rad.set_color(color, 0, 0)
                self.poitrans = rendering.Transform()
                self.radtrans = rendering.Transform()
                poi.add_attr(self.poitrans)
                observation_rad.add_attr(self.radtrans)
                self.viewer.add_geom(poi)
                self.viewer.add_geom(observation_rad)
                # Scale POI location to screen size and place POI
                poi_x = self.poi_loc['poi_' + str(i)]['loc'][0]*scale+25
                poi_y = self.poi_loc['poi_' + str(i)]['loc'][1]*scale+25
                self.poitrans.set_translation(poi_x, poi_y)
                self.radtrans.set_translation(poi_x, poi_y)

        # Dict to store vertices to draw agents' paths
        path = {}

        color = 0
        for a in range(len(self.agent_trans)):
            # Scale agent location to screen size and place agent
            agent_x = self.agent_loc['agent_' + str(a)]['loc'][0]*scale+25
            agent_y = self.agent_loc['agent_' + str(a)]['loc'][1]*scale+25
            self.agent_trans[a].set_translation(agent_x, agent_y)

            # Render agent's path
            path["agent" + str(a)] = []
            for loc in self.path["agent" + str(a)]:
                x = loc[0] * scale+25
                y = loc[1] * scale+25
                path["agent" + str(a)].append(tuple([x, y]))
            agent_path = rendering.make_polyline(path["agent" + str(a)])
            color = color + (1 / self.num_agents)
            agent_path.set_color(0, color, 0)   #rgb
            self.viewer.add_geom(agent_path)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


    def close(self):
        if self.viewer: self.viewer.close()