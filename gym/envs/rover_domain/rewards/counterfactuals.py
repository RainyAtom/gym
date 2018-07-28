from math import sqrt
from gym.envs.rover_domain.rewards.g import GlobalReward
from gym.envs.rover_domain.rewards.d import DifferenceReward

def cf(domain_state, agent_id, agent_info, coupling=1, consideration_radius=5.0):
    """
    Implement counterfactual and return new state of the domain
    :param domain_state: current state of the domain
    :param agent_id: id of agent considering counterfactual agents
    :param agent_info: info of agent considering counterfactual agents, includes location
    :param coupling: the number of agents that must observe different POIs for max reward
    :param consideration_radius: the maximum distance from an agent POIs must be to have a counterfactual agent added
    :returns: domain state with counterfactual implemented
    """
    # Domain to implement counterfactual
    new_domain_state = domain_state
    # list of POIs in range of agent
    considered_poi = []

    # For every POI in the domain, determine if it is within range of the agent
    for poi_id, poi_info in domain_state['pois'].items():
        dist = GlobalReward.distance(agent_info['loc'], poi_info['loc'])
        # If the distance between the agent and POI is within the consideration radius, store the POI's location
        if dist <= consideration_radius:
            considered_poi.append(poi_info['loc'])


    # If no POI in range, return current domain state
    if len(considered_poi) is 0:
        return domain_state

    # list of POIs that with an additional agent could increase reward
    cf_poi = []
    # Calculate difference reward
    agent_diff = DifferenceReward.calculate_reward(domain_state, agent_id)
    # Calculate D++ with additional agents placed at each of the POIs in range of the agent
    cf_diff = cf_D(domain_state, domain_state, considered_poi)

    # Compare D++ evaluations to see if counterfactual agents improve reward
    if cf_diff <= agent_diff:
        return domain_state
    else:
        # for each POI in range of the agent, place an additional agent and check for reward improvement
        for poi_loc in considered_poi and len(cf_poi) < coupling - 1:
            # Calculate D++ with additional agent placed at POI considered
            cf_diff = cf_D(domain_state, domain_state, poi_loc)

            if cf_diff > agent_diff:
                cf_poi.append(poi_loc)

    # create a new agent at each POI that improved reward
    for poi_loc in cf_poi:
        new_domain_state['agents'].append({'loc': poi_loc, 'theta': 0})

    return new_domain_state

def cf_D(temp_domain, domain_state, considered_poi):
    # Place additional agents at each of the POIs in range of the agent
    for poi_loc in considered_poi:
        temp_domain['agents'].append({'loc': poi_loc, 'theta': 0})
    # Calculate D++ with additional agents placed at each of the POIs in range of the agent
    cf_diff = GlobalReward.calculate_reward(temp_domain) - GlobalReward.calculate_reward(domain_state)
    cf_diff = cf_diff / len(considered_poi)

    return cf_diff

