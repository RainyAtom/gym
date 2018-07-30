import gym.spaces
from gym import spaces
from gym.utils import seeding
from gym.envs.rover_domain.simulators.rover_domain_simulator import RoverDomainS
from gym.envs.classic_control import rendering
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

        if config_f is None:
            config_f = "config.yml"
        with open(config_f, 'r') as f:
            config = yaml.load(f)

        # Initialize the rover domain.
        self.domain = RoverDomainS(
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

        self.path = {}
        for a in range(self.num_agents):
            self.path["agent" + str(a)] = [tuple(self.agent_loc['agent_' + str(a)]['loc'])]

        self.viewer = None

        # self.action_space = spaces.Discrete(self.world_width)
        # self.observation_space = self.domain.get_jointstate()
        self.seed()
        self.reset()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, team):
        #print("Path: " + str(self.path))

        # Get States from Rover Doman
        joint_state = self.domain.get_jointstate()
        # Get the actions from the team
        actions = team.get_jointaction(joint_state)
        # Pass actions to domain to update
        self.domain.apply_actions(actions)
        # Update the joint state
        joint_state = self.domain.get_jointstate()

        for i in range(self.num_agents):
            self.path["agent" + str(i)].append(tuple(joint_state['agents']['agent_' + str(i)]['loc']))

        done = True # serves no purpose atm

        return joint_state, 0, done, {}


    def reset(self):
        return self.domain.get_jointstate()


    def render(self, mode='human'):

        screen_width = 500
        screen_height = 500
        scale = screen_width / self.world_width
        agent_side = 10
        poi_rad = 5
        path = {}

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.agent_trans = []

            # Initialize agents
            color = 0
            for i in range(self.num_agents):
                color = color + (1/self.num_agents)
                l, r, t, b = -agent_side/2, agent_side/2, agent_side/2, -agent_side/2   # left, right, top, bottom
                agent = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
                agent.set_color(0, color, 0)
                agenttrans = rendering.Transform()
                agent.add_attr(agenttrans)

                self.agent_trans.append(agenttrans)
                self.viewer.add_geom(agent)

            # Initialize and Render POIs and their observation radius
            color = 0
            for i in range(self.num_pois):
                color = color + (1/self.num_pois)
                poi = rendering.make_circle(poi_rad)
                observation_rad = rendering.make_circle(self.observation_rad*scale, 30, False)
                poi.set_color(color, 0, 0)  #rgb
                observation_rad.set_color(color, 0, 0)
                self.poitrans = rendering.Transform()
                self.radtrans = rendering.Transform()
                poi.add_attr(self.poitrans)
                observation_rad.add_attr(self.radtrans)
                self.viewer.add_geom(poi)
                self.viewer.add_geom(observation_rad)

                poi_x = self.poi_loc['poi_' + str(i)]['loc'][0]*scale
                poi_y = self.poi_loc['poi_' + str(i)]['loc'][1]*scale
                self.poitrans.set_translation(poi_x, poi_y)
                self.radtrans.set_translation(poi_x, poi_y)

        color = 0
        for a in range(len(self.agent_trans)):
            # Render agents
            agent_x = self.agent_loc['agent_' + str(a)]['loc'][0]*scale
            agent_y = self.agent_loc['agent_' + str(a)]['loc'][1]*scale
            self.agent_trans[a].set_translation(agent_x, agent_y)

            # Render agent's path
            color = color + (1 / self.num_agents)
            path["agent" + str(a)] = []
            for loc in self.path["agent" + str(a)]:
                x = loc[0] * scale
                y = loc[1] * scale
                path["agent" + str(a)].append(tuple([x, y]))
            agent_path = rendering.make_polyline(path["agent" + str(a)])
            agent_path.set_color(0, color, 0)
            self.viewer.add_geom(agent_path)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


    def close(self):
        if self.viewer: self.viewer.close()


