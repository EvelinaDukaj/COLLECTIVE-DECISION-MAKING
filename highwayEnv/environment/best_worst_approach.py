import pickle
import pandas as pd


def agent_Dictionary(id, dictionary):

    """
    Create a dictionary for every agent from the big dictionary.
    """

    agent_dictionary = {'hypervolume': [], 'first_risk': [], 'counter': [], 'threshold': [], 'cumWaitTime': [],
                        'cumCrashes': [], 'avgFitness': [], 'time_nd': [], 'risk_nd': [], 'case': [], 'id': []}
    for i in range(len(dictionary['id'])):
        if id == int(dictionary['id'][i]):
            agent_dictionary['hypervolume'].append(dictionary['hypervolume'][i])
            agent_dictionary['first_risk'].append(dictionary['first_risk'][i])
            agent_dictionary['counter'].append(dictionary['counter'][i])
            agent_dictionary['threshold'].append(dictionary['threshold'][i])
            agent_dictionary['cumWaitTime'].append(dictionary['cumWaitTime'][i])
            agent_dictionary['cumCrashes'].append(dictionary['cumCrashes'][i])
            agent_dictionary['avgFitness'].append(dictionary['avgFitness'][i])
            agent_dictionary['time_nd'].append(dictionary['time_nd'][i])
            agent_dictionary['risk_nd'].append(dictionary['risk_nd'][i])
            agent_dictionary['case'].append(dictionary['case'][i])
            agent_dictionary['id'].append(dictionary['id'][i])

    return agent_dictionary


def find_best_fitness(agent_dictionary):
    """
    Find the best fitness (lowest value) in the selected agent dictionary.
    """

    agent_new = []
    best_fitness = min(agent_dictionary['avgFitness'])
    for i in range(len(agent_dictionary['avgFitness'])):
        if best_fitness == agent_dictionary['avgFitness'][i]:
            agent_new = [agent_dictionary['hypervolume'][i],
                         agent_dictionary['first_risk'][i],
                         agent_dictionary['counter'][i],
                         agent_dictionary['threshold'][i],
                         agent_dictionary['cumWaitTime'][i],
                         agent_dictionary['cumCrashes'][i],
                         agent_dictionary['avgFitness'][i],
                         agent_dictionary['time_nd'][i],
                         agent_dictionary['risk_nd'][i],
                         agent_dictionary['case'][i]]

    return agent_new


def find_worst_fitness(agent_dictionary):
    """
        Find the worst fitness (highest value) in the selected agent dictionary.
        """

    worst_fitness = max(agent_dictionary['avgFitness'])
    return worst_fitness


def switch_values(id_best, id_worst, dictionary):
    """
    Replace the attribute values of the agent with the worst fitness with the values of the agent
    with the best fitness except for the id.
    """

    agent1Dic = agent_Dictionary(id_worst, dictionary)
    agent2Dic = agent_Dictionary(id_best, dictionary)

    a1_worst = find_worst_fitness(agent1Dic)
    a2_best = find_best_fitness(agent2Dic)

    for i in range(len(dictionary['avgFitness'])):
        if id_worst == int(dictionary['id'][i]):
            if float(dictionary['avgFitness'][i]) == a1_worst:
               dictionary['hypervolume'][i] = a2_best[0]
               dictionary['first_risk'][i] = a2_best[1]
               dictionary['counter'][i] = a2_best[2]
               dictionary['threshold'][i] = a2_best[3]
               dictionary['cumWaitTime'][i] = a2_best[4]
               dictionary['cumCrashes'][i] = a2_best[5]
               dictionary['avgFitness'][i] = a2_best[6]
               dictionary['time_nd'][i] = a2_best[7]
               dictionary['risk_nd'][i] = a2_best[8]
               dictionary['case'][i] = a2_best[9]
               break
