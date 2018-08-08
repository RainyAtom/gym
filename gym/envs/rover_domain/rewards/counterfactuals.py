from gym.envs.rover_domain.rewards.g import GlobalReward
from gym.envs.rover_domain.rewards.d import DifferenceReward

# TODO: this will need to be modified to fit within a D++ implementation

def cf_init(cf, domain_state, agent_id, agent_info, consideration_radius=5.0):
    """
    Determine if there are POIs within range of the agent, if there are then place counterfactual agents one of two ways
    :param cf: indicates counterfactual implementation version
    :param domain_state: current state of the domain
    :param agent_id: id of agent considering counterfactual agents
    :param agent_info: info of agent considering counterfactual agents, location
    :param consideration_radius: the maximum distance from an agent POIs must be to have a counterfactual agent added
    :returns: domain state with counterfactual implemented
    """
    # list of POIs in range of agent
    considered_poi = []

    # For every POI in the domain, determine if it is within range of the agent
    for poi_id, poi_info in domain_state['pois'].items():
        dist = GlobalReward.distance(agent_info['loc'], poi_info['loc'])
        # If the distance between the agent and POI is within the consideration radius, store the POI's location
        if dist <= consideration_radius:
            considered_poi.append(poi_info['loc'])

    # If no POI in range, return current domain state
    if len(considered_poi) == 0:
        return domain_state
    else:
        # Calculate difference reward
        agent_diff = DifferenceReward.calculate_reward(domain_state, agent_id)
        # Calculate D++ with additional agents placed at each of the POIs in range of the agent
        cf_diff = cf_D(domain_state, considered_poi)

        # Compare D++ evaluations to see if counterfactual agents improve reward
        if cf_diff <= agent_diff:
            return domain_state
        else:
            if cf == 0:
                single_agent(domain_state, considered_poi, agent_diff)
            elif cf == 1:
                multi_agent(domain_state, considered_poi, agent_diff)


def single_agent(domain_state, considered_poi, agent_diff):
    """
    D++ Extension: Place a single counterfactual agent at each POI within range
    :param domain_state: current state of the domain
    :param considered_poi: list of POI within range of agent
    :param agent_diff: difference reward
    :returns: domain state with counterfactual implemented
    """
    # Domain to implement counterfactual
    new_domain_state = domain_state
    # number of counterfactual agents that can be added
    agent_limit = len(domain_state["agents"])- 1
    # list of POIs that with an additional agent could increase reward
    cf_poi = []

    # for each POI in range of the agent, place an additional agent and check for reward improvement
    for poi_loc in considered_poi and len(cf_poi) < agent_limit:
        # Calculate D++ with additional agent placed at POI considered
        cf_diff = cf_D(domain_state, poi_loc)
        # Checks if additional agent improves reward
        if cf_diff > agent_diff:    # NOTE: not sure if this is the right comparison
            cf_poi.append(poi_loc)
            # create a new agent at POI that improved reward
            new_domain_state['agents'].append({'loc': poi_loc, 'theta': 0})  # NOTE: may need to indicate a specific name "agent_" + num

    return new_domain_state


def multi_agent(domain_state, considered_poi, agent_diff):
    """
    D++ Extension: Place multiple counterfactual agents (distribute team size) at each POI within range
    :param domain_state: current state of the domain
    :param considered_poi: list of POI within range of agent
    :param agent_diff: difference reward
    :returns: domain state with counterfactual implemented
    """
    # Domain to implement counterfactual
    new_domain_state = domain_state
    # number of counterfactual agents that can be added
    agent_limit = len(domain_state["agents"])- 1
    compare_d = agent_diff

    # list of POIs that with an additional agent could increase reward
    cf_poi = []

    while(agent_limit is not 0):
        # for each POI in range of the agent, place an additional agent and check for reward improvement
        for poi_loc in considered_poi and len(cf_poi) < agent_limit:
            # Calculate D++ with additional agent placed at POI considered
            cf_diff = cf_D(new_domain_state, poi_loc) # NOTE: not sure if this is right
            # Checks if additional agent improves reward
            if cf_diff > compare_d:    # NOTE: not sure if this is the right comparison
                cf_poi.append(poi_loc)
                # create a new agent at POI that improved reward
                new_domain_state['agents'].append({'loc': poi_loc, 'theta': 0})  # NOTE: may need to indicate a specific name "agent_" + num
        # this is probs not right
        compare_d = cf_D(domain_state, cf_poi)
        domain_state = new_domain_state

        agent_limit = agent_limit - len(cf_poi)
        considered_poi = cf_poi
        cf_poi = []

    return new_domain_state


def cf_D(domain_state, considered_poi):
    temp_domain = domain_state
    # Place additional agents at each of the POIs in range of the agent
    for poi_loc in considered_poi:
        temp_domain['agents'].append({'loc': poi_loc, 'theta': 0})  # may need to indicate a specific name "agent_" + num
    # Calculate D++ with additional agents placed at each of the POIs in range of the agent
    cf_diff = GlobalReward.calculate_reward(temp_domain) - GlobalReward.calculate_reward(domain_state)
    cf_diff = cf_diff / len(considered_poi)

    return cf_diff