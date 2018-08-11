from gym.envs.rover_domain.rewards.g import GlobalReward
from gym.envs.rover_domain.rewards.d import DifferenceReward

"""
TODO: Needs to be modified to fit within a D++ implementation in a different version of the rover domain.
    -Names for accessing domain_state information will need to be adjusted.
"""

def cf(cf, domain_state, agent_id, agent_info, consideration_radius=5.0):
    """
    If there are POIs within range of the agent, then place counterfactual agents
    :param cf: indicates counterfactual implementation version, 0-one by one 1-waves
    :param domain_state: current state of the domain
    :param agent_id: id of agent considering counterfactual agents
    :param agent_info: info of agent considering counterfactual agents, location
    :param consideration_radius: the maximum distance from an agent POIs must be to have a counterfactual agents added
    :returns: domain state with counterfactual implemented
    """
    # List of POIs in range of agent
    considered_poi = []

    # For every POI in the domain, determine if it is within range of the agent
    for poi_id, poi_info in domain_state['pois'].items():
        dist = GlobalReward.distance(agent_info['loc'], poi_info['loc'])
        # If the distance between the agent and POI is within the consideration radius, store the POI's location
        if dist <= consideration_radius:
            considered_poi.append(poi_info['loc'])

    # Number of counterfactual agents that can be added
    agent_limit = len(domain_state["agents"]) - 1

    # If no POIs in range or if team size is 1, return current domain state
    if len(considered_poi) == 0 or agent_limit == 0:
        return domain_state
    # Else add counterfactual agents
    else:
        # Calculate difference reward
        compare_d = DifferenceReward.calculate_reward(domain_state, agent_id)
        # Calculate D++ with team size distributed between each of the POIs in range of the agent
        cf_diff = modified_cf_D(domain_state, considered_poi, agent_limit)

        # Compare evaluations to see if counterfactual agents improve reward
        if cf_diff <= compare_d:
            return domain_state
        else:
            # bool flag
            reward_improved = False
            # Domain to implement counterfactual agents
            new_domain_state = domain_state
            # new agent id
            id_num = len(domain_state["agents"])
            # List of POIs where counterfactual agents were placed
            cf_poi = []

            # Place counterfactual agents one by one
            if cf == 0:
                while (reward_improved == False):
                    for poi_loc in considered_poi:
                        # Calculate D++ with additional agent placed at POI considered
                        cf_diff = cf_D(new_domain_state, poi_loc)
                        # Checks if additional agent improves reward
                        if cf_diff <= compare_d:
                            cf_poi.append(poi_loc)
                            # create a new agent at POI that DIDN'T improve reward
                            new_agent_id = "agent_" + str(id_num)
                            id_num = id_num + 1
                            new_domain_state['agents'][new_agent_id] = {'loc': poi_loc, 'theta': 0}
                        else:
                            cf_poi.append(poi_loc)
                            # create a new agent at POI that improved reward
                            new_agent_id = "agent_" + str(id_num)
                            id_num = id_num + 1
                            new_domain_state['agents'][new_agent_id] = {'loc': poi_loc, 'theta': 0}
                            reward_improved = True
                            break
            # Place counterfactual agents in waves. wave --> one agent per POI
            elif cf == 1:
                while (reward_improved == False):
                    for poi_loc in considered_poi and len(cf_poi) < agent_limit:
                        cf_poi.append(poi_loc)
                        # create a new agent at POI
                        new_agent_id = "agent_" + str(id_num)
                        id_num = id_num + 1
                        new_domain_state['agents'][new_agent_id] = {'loc': poi_loc, 'theta': 0}

                    cf_diff = cf_D(domain_state, cf_poi)
                    if cf_diff > compare_d:
                        reward_improved = True

            return new_domain_state


def cf_D(domain_state, considered_poi):
    """
    Placeholder for D++ evaluation
    :param domain_state: current state of the domain
    :param considered_poi: where to place counterfactual agents
    :return:
    """
    temp_domain = domain_state
    id = len(domain_state['agents'])
    # Place additional agents at each of the POIs in range of the agent
    for poi_loc in considered_poi:
        new_agent_id = "agent_" + str(id)
        id = id + 1
        temp_domain['agents'][new_agent_id] = {'loc': poi_loc, 'theta': 0}

    cf_diff = GlobalReward.calculate_reward(temp_domain) - GlobalReward.calculate_reward(domain_state)
    cf_diff = cf_diff / len(considered_poi)

    return cf_diff


def modified_cf_D(domain_state, considered_poi, agent_limit):
    """
    Placeholder for D++ evaluation
    :param domain_state: current state of the domain
    :param considered_poi: where to place counterfactual agents
    :param agent_limit: max number of counterfactual agents
    :return:
    """
    # modified cf_D(domain_state, considered_poi)
    temp_domain = domain_state
    id = len(domain_state['agents'])
    # Place additional agents at each of the POIs in range of the agent
    while (agent_limit is not 0):
        for poi_loc in considered_poi:
            if agent_limit is not 0:
                new_agent_id = "agent_" + str(id)
                id = id + 1
                temp_domain['agents'][new_agent_id] = {'loc': poi_loc, 'theta': 0}
                agent_limit = agent_limit - 1
            else:
                break

    cf_diff = GlobalReward.calculate_reward(temp_domain) - GlobalReward.calculate_reward(domain_state)
    cf_diff = cf_diff / len(considered_poi)

    return cf_diff