import gym
import yaml
import sys
import time
from gym.envs.rover_domain.teams.rover_team import RoverTeam
from gym.envs.rover_domain.policies.policy import RandomPolicy
from gym.envs.rover_domain.policies.policy import CCEA
from gym.envs.rover_domain.policies.policy import Evo_MLP
from gym.envs.rover_domain.rewards.g import GlobalReward
import pyglet

def main(config_f):
    # Unique identifier for each run, used in naming files
    id = str(time.clock())

    env = gym.make('rover-v0')
    observation = env.reset()

    # Read and store parameters from configuration file.
    if config_f is None:
        config_f = "config.yml"
    with open(config_f, 'r') as f:
        config = yaml.load(f)

    # # Write config file name to file meant to store reward
    # with open(id + '_global_reward.yml', 'a') as file:
    #     file.write(config_f + "\n")

    # Initialize agent policies
    agent_policies = {}
    for i in range(config["Number of Agents"]):
        agent_policies["agent_"+str(i)] = Evo_MLP(8, 2)
    team = RoverTeam(agent_policies)

    # Initialize the reward function
    global_reward = GlobalReward(
        config["Coupling"],
        config["Observation Radius"],
        config["Minimum Distance"])

    for generation in range(config["Epochs"]):
        for step in range(config["Steps"]):
            env.render()
            observation, reward, done, info = env.step(team)
            # reward.record_history(observation)

        print("Generation " + str(generation+1) + ": " + str(observation))

        # # Compute the Global Reward
        # fitness = global_reward.calculate_reward()
        #
        # # Store the global reward in a file
        # with open(id + '_global_reward.yml', 'a') as file:
        #     file.write(str(fitness['agent_0']) + "\n")
        #
        # # CCEA Evaluation
        # CCEA(team, fitness)

    # screenshot domain rendering when done
    pyglet.image.get_buffer_manager().get_color_buffer().save(id + config_f + '.png')


if __name__ == '__main__':
    # When ran through command line and no specific file is indicated, use default configuration file
    if len(sys.argv) is 1:
        config_f = "config.yml"
    else:
        config_f = sys.argv[1]
    main(config_f)