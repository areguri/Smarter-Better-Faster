import copy
import argparse
import numpy as np
import random
import sys
import os
import math
import pandas as pd

sys.path.append(os.path.abspath(__file__))
from environment import *
from utils import *
from models import *
inf = math.inf


class Agent(The_Environment) :
    def __init__(self) :
        pass

    def u_star_agent(self, env, utility, policy) :
        agent_loc = env.agent_location
        prey_loc = env.prey_location
        pred_loc = env.predator_location
        prey_dict = {}
        pred_dict = {}
        ag_dict = {}
        edges = env.edges
        step_count = 0
        while (True) :
            # print("Agent, prey and predator locations are ", agent_loc, prey_loc, pred_loc)
            initial_agent_loc = agent_loc
            initial_prey_loc = prey_loc
            initial_pred_loc = pred_loc
            if (prey_loc == agent_loc) :
                # print("Agent caught the prey")
                return True, False, step_count, ag_dict, prey_dict, pred_dict
            if (agent_loc == pred_loc) :
                # print("Predator caught the agent")
                return False, False, step_count, ag_dict, prey_dict, pred_dict
            agent_choices = edges[agent_loc]
            # print("Agent choices ", agent_choices)
            local_utility = {}
            for agent_choice in agent_choices :
                local_utility[agent_choice, prey_loc, pred_loc] = utility[agent_choice, prey_loc, pred_loc]
            local_utility = {k : v for k, v in sorted(local_utility.items(), key=lambda item : item[1])}
            # print("Local utilities are ", local_utility)
            agent_loc = policy[agent_loc, prey_loc, pred_loc]
            # print("Agent location is ", agent_loc)
            step_count = step_count + 1
            if (agent_loc == pred_loc) :
                return False, False, step_count, ag_dict, prey_dict, pred_dict
            if (prey_loc == agent_loc) :
                return True, False, step_count, ag_dict, prey_dict, pred_dict
            prey_loc = move_prey(env, prey_loc)
            pred_loc = easily_distracted_predator(env, pred_loc, agent_loc)

            ag_dict[initial_agent_loc, initial_prey_loc, initial_pred_loc] = agent_loc
            pred_dict[initial_agent_loc, initial_prey_loc, initial_pred_loc] = pred_loc
            prey_dict[initial_agent_loc, initial_prey_loc, initial_pred_loc] = prey_loc

    def v_star_agent(self, env, utility) :
        agent_loc = env.agent_location
        prey_loc = env.prey_location
        pred_loc = env.predator_location
        prey_dict = {}
        pred_dict = {}
        ag_dict = {}
        edges = env.edges
        step_count = 0
        while (True) :
            # print("Agent, prey and predator locations are ", agent_loc, prey_loc, pred_loc)
            initial_agent_loc = agent_loc
            initial_prey_loc = prey_loc
            initial_pred_loc = pred_loc
            if (prey_loc == agent_loc) :
                # print("Agent caught the prey")
                return True, False, step_count, ag_dict, prey_dict, pred_dict
            if (agent_loc == pred_loc) :
                # print("Predator caught the agent")
                return False, False, step_count, ag_dict, prey_dict, pred_dict

            action_utils = {}
            prey_probs = {}
            for new_pr in edges[prey_loc] + [prey_loc] :
                prey_probs[new_pr] = 1 / (len(edges[prey_loc]) + 1)
            pred_probs = compute_predator_probabilities(env, pred_loc, agent_loc)
            for new_ag in edges[agent_loc] :
                if (new_ag == prey_loc) and (new_ag != pred_loc):
                    action_utils[new_ag] = 0
                    continue
                if new_ag == pred_loc :
                    action_utils[new_ag] = inf
                    continue
                sum = 0
                for new_pr in edges[prey_loc] + [prey_loc] :
                    for new_pred in edges[pred_loc] :
                        sum = sum + prey_probs[new_pr] * pred_probs[new_pred] * utility[new_ag, new_pr, new_pred]
                action_utils[new_ag] = sum

            action_utils = {k : v for k, v in sorted(action_utils.items(), key=lambda item : item[1])}
            agent_loc = list(action_utils.keys())[0]
            # print("Agent location is ", agent_loc)
            step_count = step_count + 1
            if agent_loc == pred_loc :
                return False, False, step_count, ag_dict, prey_dict, pred_dict
            if prey_loc == agent_loc :
                return True, False, step_count, ag_dict, prey_dict, pred_dict
            prey_loc = move_prey(env, prey_loc)
            pred_loc = easily_distracted_predator(env, pred_loc, agent_loc)

            ag_dict[initial_agent_loc, initial_prey_loc, initial_pred_loc] = agent_loc
            pred_dict[initial_agent_loc, initial_prey_loc, initial_pred_loc] = pred_loc
            prey_dict[initial_agent_loc, initial_prey_loc, initial_pred_loc] = prey_loc

    def agent_1(self, env, init_step_count) :
        prey_dict = {}
        pred_dict = {}
        ag_dict = {}

        agent_loc = env.agent_location
        prey_loc = env.prey_location
        pred_loc = env.predator_location

        edges = env.edges
        step_count = 0
        while (True) :
            initial_agent_loc = agent_loc
            initial_prey_loc = prey_loc
            initial_pred_loc = pred_loc

            if (prey_loc == agent_loc) :
                return True, False, step_count, ag_dict, prey_dict, pred_dict
            if (agent_loc == pred_loc) :
                return False, False, step_count, ag_dict, prey_dict, pred_dict
            agent_choices = edges[agent_loc]
            if (prey_loc in agent_choices and prey_loc != pred_loc) :
                # print("found the prey in neighbour")
                agent_loc = prey_loc
            else :
                pred_distance = get_distance(agent_loc, pred_loc, edges)
                prey_distance = get_distance(agent_loc, prey_loc, edges)
                prey_locs = {}
                pred_locs = {}
                for choice in agent_choices :
                    prey_locs[choice] = get_distance(choice, prey_loc, edges)
                    pred_locs[choice] = get_distance(choice, pred_loc, edges)
                agent_loc = get_full_information_choice(prey_locs, pred_locs, prey_distance, pred_distance, agent_loc)
            step_count = step_count + 1
            if (step_count == init_step_count) :
                return False, True, 0, ag_dict, prey_dict, pred_dict
            if (agent_loc == pred_loc) :
                return False, False, step_count, ag_dict, prey_dict, pred_dict
            if (prey_loc == agent_loc) :
                return True, False, step_count, ag_dict, prey_dict, pred_dict
            prey_loc = move_prey(env, prey_loc)
            pred_loc = easily_distracted_predator(env, pred_loc, agent_loc)
            ag_dict[initial_agent_loc, initial_prey_loc, initial_pred_loc] = agent_loc
            pred_dict[initial_agent_loc, initial_prey_loc, initial_pred_loc] = pred_loc
            prey_dict[initial_agent_loc, initial_prey_loc, initial_pred_loc] = prey_loc

    def agent_2(self, env, init_step_count) :
        prey_dict = {}
        pred_dict = {}
        ag_dict = {}

        agent_loc = env.agent_location
        prey_loc = env.prey_location
        pred_loc = env.predator_location
        edges = copy.deepcopy(env.edges)
        step_count = 0
        while (True) :
            initial_agent_loc = agent_loc
            initial_prey_loc = prey_loc
            initial_pred_loc = pred_loc

            if (prey_loc == agent_loc) :
                # print("Agent 2 caught the prey")
                return True, False, step_count, ag_dict, prey_dict, pred_dict
            if (agent_loc == pred_loc) :
                # print("Predator caught Agent 2")
                return False, False, step_count, ag_dict, prey_dict, pred_dict
            agent_choices = edges[agent_loc]
            if prey_loc in agent_choices and prey_loc != pred_loc:
                # print("Agent 2 found the prey in neighbour")
                agent_loc = prey_loc
            else :
                pred_distance = get_distance(agent_loc, pred_loc, edges)
                prey_distance = get_distance(agent_loc, prey_loc, edges)
                prey_locs = {}
                pred_locs = {}
                for choice in agent_choices :
                    distance = 0
                    if get_distance(choice, pred_loc, edges) < 2 :
                        continue
                    prey_neighbours = edges[prey_loc]
                    for prey_neighbr in prey_neighbours :
                        distance += get_distance(choice, prey_neighbr, edges)
                    prey_locs[choice] = distance / len(prey_neighbours)
                    pred_locs[choice] = get_distance(choice, pred_loc, edges)
                if len(prey_locs) > 0 :
                    agent_loc = get_full_information_choice(prey_locs, pred_locs, prey_distance, pred_distance,
                                                            agent_loc)
            step_count = step_count + 1
            if (step_count == init_step_count) :
                return False, True, 0, ag_dict, prey_dict, pred_dict
            if (agent_loc == pred_loc) :
                # print("Predator caught the agent")
                return False, False, step_count, ag_dict, prey_dict, pred_dict
            if (prey_loc == agent_loc) :
                # print("Agent caught the prey")
                return True, False, step_count, ag_dict, prey_dict, pred_dict
            prey_loc = move_prey(env, prey_loc)
            pred_loc = easily_distracted_predator(env, pred_loc, agent_loc)

            ag_dict[initial_agent_loc, initial_prey_loc, initial_pred_loc] = agent_loc
            pred_dict[initial_agent_loc, initial_prey_loc, initial_pred_loc] = pred_loc
            prey_dict[initial_agent_loc, initial_prey_loc, initial_pred_loc] = prey_loc

    def v_partial(self, env):
        agent_loc = env.agent_location
        true_prey_loc = env.prey_location
        prey_loc = None
        pred_loc = env.predator_location
        edges = env.edges
        step_count = 0
        prey_prob = init_prey_probs(agent_loc)
        
        indx = 0
        while(True):
          if(agent_loc == pred_loc):
            return False, False, step_count
          if(true_prey_loc == agent_loc):
            return True, False, step_count
         
          survey_node = choose_node_for_survey(prey_prob)
          if survey_node:
            prey_prob = update_prey_probs_by_survey(prey_prob, true_prey_loc, survey_node)
          
          agent_choices = edges[agent_loc]
          local_utility = {}

          for agent_choice in agent_choices:
            ag_prey_distance = calculate_ag_prey_distance(edges, agent_choice, prey_prob)
            if agent_choice == pred_loc : 
              local_utility[agent_choice, pred_loc] = inf
              continue
            elif (agent_choice == true_prey_loc):
                local_utility[agent_choice, pred_loc] = 0
            else:
              pred_prob = compute_predator_probabilities(env,pred_loc,agent_choice)
              lu = 0
              for pred_choice in edges[pred_loc]: 
                agent_pred_distance = get_distance(agent_choice, pred_choice, edges)
                if (agent_choice == pred_choice):
                  lu = lu+inf
                else :
                  u_part = predict_model_v_partial(agent_choice, pred_choice, prey_prob, ag_prey_distance, agent_pred_distance)
                  lu = lu+ pred_prob[pred_choice]*u_part
              local_utility[agent_choice, pred_loc] = lu
          
          local_utility = {k: v for k, v in sorted(local_utility.items(), key=lambda item: item[1])}
          agent_loc = list(local_utility.keys())[0][0]
          prey_prob = update_prey_probs_by_agent(prey_prob, true_prey_loc, agent_loc)
          step_count = step_count + 1
          if(agent_loc == pred_loc):
              return False, False, step_count
          if(true_prey_loc == agent_loc):
              return True, False, step_count
          true_prey_loc = move_prey(env, true_prey_loc)
          prey_prob = update_prey_probs_by_prey_movement(prey_prob, edges)
          pred_loc = easily_distracted_predator(env, pred_loc, agent_loc)

    def u_partial(self, env, utility) :
        agent_loc = env.agent_location
        true_prey_loc = env.prey_location
        prey_loc = None
        pred_loc = env.predator_location
        edges = env.edges
        step_count = 0
        prey_prob = init_prey_probs(agent_loc)
        feature_data = pd.DataFrame(
            columns=['agent_loc', 'pred_loc', 'prey_prob', 'ag_pred_dist', 'ag_prey_dist', 'policy', 'policy_prey',
                     'policy_pred', 'u_partial_ag_loc'])

        indx = 0
        while (True) :
            # print("Agent, prey and predator locations are ", agent_loc, true_prey_loc, pred_loc)
            if agent_loc == pred_loc :
                # print("Predator caught the agent u_partial")
                return False, False, step_count, feature_data
            if true_prey_loc == agent_loc :
                # print("agent u_partial caught the prey")
                return True, False, step_count, feature_data

            survey_node = choose_node_for_survey(prey_prob)
            if survey_node :
                prey_prob = update_prey_probs_by_survey(prey_prob, true_prey_loc, survey_node)

            agent_choices = edges[agent_loc]
            local_utility = {}
            ag_pred = get_distance(agent_loc, pred_loc, env.edges)
            ag_prey = calculate_ag_prey_distance(env.edges, agent_loc, prey_prob)
            u_partial_temp = calculate_u_partial(utility, agent_loc, pred_loc, prey_prob)

            for agent_choice in agent_choices :
                if agent_choice == pred_loc :
                    local_utility[agent_choice, pred_loc] = inf
                    continue
                elif (agent_choice == true_prey_loc) :
                    local_utility[agent_choice, pred_loc] = 0
                else :
                    pred_prob = compute_predator_probabilities(env, pred_loc, agent_choice)
                    lu = 0
                    for pred_choice in edges[pred_loc] :
                        if (agent_choice == pred_choice) :
                            lu = lu + inf
                        else :
                            u_part = calculate_u_partial(utility, agent_choice, pred_choice, prey_prob)
                            lu = lu + pred_prob[pred_choice] * u_part
                    local_utility[agent_choice, pred_loc] = lu

            local_utility = {k : v for k, v in sorted(local_utility.items(), key=lambda item : item[1])}
            # init_ag_loc = agent_loc
            # initial_prey_prb = prey_prob
            agent_loc = list(local_utility.keys())[0][0]
            prey_prob = update_prey_probs_by_agent(prey_prob, true_prey_loc, agent_loc)
            # policy_pred =  get_distance(agent_loc, pred_loc, env.edges)
            # policy_prey = calculate_ag_prey_distance(env.edges, agent_loc, prey_prob)
            # feature_data.loc[indx] = [init_ag_loc, pred_loc, initial_prey_prb, ag_pred, ag_prey,list(local_utility.keys())[0][0],policy_prey,policy_pred,u_partial_temp]
            # indx+=1
            step_count = step_count + 1
            if (agent_loc == pred_loc) :
                # print("Pred caught the Agent u_partial")
                return False, False, step_count, feature_data
            if (true_prey_loc == agent_loc) :
                # print("Agent u_partial caught the prey")
                return True, False, step_count, feature_data
            true_prey_loc = move_prey(env, true_prey_loc)
            prey_prob = update_prey_probs_by_prey_movement(prey_prob, edges)
            # ag_pred =  get_distance(agent_loc, pred_loc, env.edges)
            # ag_prey = calculate_ag_prey_distance(  env.edges, agent_loc, prey_prob)
            # u_partial_temp = calculate_u_partial(utility, agent_loc, pred_loc, prey_prob)
            # policy_prey = calculate_ag_prey_distance( env.edges, agent_loc, prey_prob)
            # #feature_data.loc[indx] = [agent_loc, pred_loc, prey_prob, ag_pred, ag_prey, agent_loc, policy_prey, policy_pred, u_partial_temp]
            # #indx+=1
            pred_loc = easily_distracted_predator(env, pred_loc, agent_loc)

    def agent_3(self, env, init_step_count) :
        agent_loc = env.agent_location
        true_prey_loc = env.prey_location
        prey_loc = None
        pred_loc = env.predator_location
        edges = env.edges
        step_count = 0
        prey_prob = init_prey_probs(agent_loc)
        while (True) :
            # print("Agent location, prey and predator location are ", agent_loc, true_prey_loc, pred_loc)
            if (true_prey_loc == agent_loc) :
                # print("Agent caught the prey")
                return True, False, step_count
            if (agent_loc == pred_loc) :
                # print("Predator caught the agent")
                return False, False, step_count
            survey_node = choose_node_for_survey(prey_prob)
            if survey_node :
                # print("Survey is chosen, updating probabilities by survey for ", survey_node)
                prey_prob = update_prey_probs_by_survey(prey_prob, true_prey_loc, survey_node)
            #     if 1 in list(prey_prob.values()) :
            #         prey_certainity["agent_3"] = prey_certainity["agent_3"] + 1
            # else :
            #     prey_certainity["agent_3"] = prey_certainity["agent_3"] + 1
            # print("prey probs after survey :", prey_prob)
            prey_loc = get_highest_prob(prey_prob)
            agent_choices = edges[agent_loc]
            pred_distance = get_distance(agent_loc, pred_loc, edges)
            prey_distance = get_distance(agent_loc, prey_loc, edges)
            prey_locs = {}
            pred_locs = {}
            for choice in agent_choices :
                prey_locs[choice] = get_distance(choice, prey_loc, edges)
                pred_locs[choice] = get_distance(choice, pred_loc, edges)
            agent_choice = get_full_information_choice(prey_locs, pred_locs, prey_distance, pred_distance, agent_loc)
            prey_prob = update_prey_probs_by_agent(prey_prob, true_prey_loc, agent_choice)
            # print("Agent choice is ", agent_choice)
            # print("prey probs after agent movement :", prey_prob)
            agent_loc = agent_choice
            step_count = step_count + 1
            if (true_prey_loc == agent_loc) :
                # print("Agent caught the prey")
                return True, False, step_count
            if (agent_loc == pred_loc) :
                # print("Predator caught the agent")
                return False, False, step_count
            if step_count == init_step_count :
                # print("Aborting since reached maxm steps")
                return False, True, 0
            prey_prob = update_prey_probs_by_prey_movement(prey_prob, edges)
            true_prey_loc = move_prey(env, true_prey_loc)
            # print("prey probs after prey movement :", prey_prob)
            # print("Sum of prey probs is ", find_sum(prey_prob))
            pred_loc = easily_distracted_predator(env, pred_loc, agent_loc)

    def agent_4(self, env, init_step_count) :
        agent_loc = env.agent_location
        true_prey_loc = env.prey_location
        prey_loc = None
        pred_loc = env.predator_location
        edges = env.edges
        step_count = 0
        prey_prob = init_prey_probs(agent_loc)
        while (True and step_count <= init_step_count) :
            # print("Agent location, prey and predator location are ", agent_loc, true_prey_loc, pred_loc)
            if (true_prey_loc == agent_loc) :
                # print("Agent caught the prey")
                return True, False, step_count
            if (agent_loc == pred_loc) :
                # print("Predator caught the agent")
                return False, False, step_count
            survey_node = choose_node_for_survey(prey_prob)
            if survey_node :
                # print("Survey is chosen, updating probabilities by survey")
                prey_prob = update_prey_probs_by_survey(prey_prob, true_prey_loc, survey_node)
            #     if 1 in list(prey_prob.values()) :
            #         prey_certainity["agent_4"] = prey_certainity["agent_4"] + 1
            # # print("prey probs after survey :", prey_prob)
            # else :
            #     prey_certainity["agent_4"] = prey_certainity["agent_4"] + 1
            prey_loc = get_highest_prob(prey_prob)
            agent_choices = edges[agent_loc]
            pred_distance = get_distance(agent_loc, pred_loc, edges)
            prey_distance = get_distance(agent_loc, prey_loc, edges)
            prey_locs = {}
            pred_locs = {}

            for choice in agent_choices :
                distance = 0
                probabolity_sum = 0

                if get_distance(choice, pred_loc, edges) < 2 :
                    continue
                prey_neighbours = edges[prey_loc]
                for prey_neighbr in prey_neighbours :
                    if prey_prob[prey_neighbr] == 0 :
                        distance += get_distance(choice, prey_neighbr, edges)
                    else :
                        distance += prey_prob[prey_neighbr] * get_distance(choice, prey_neighbr, edges)
                    probabolity_sum += prey_prob[prey_neighbr]
                if probabolity_sum == 0 :
                    prey_locs[choice] = distance / len(prey_neighbours)
                else :
                    prey_locs[choice] = distance / probabolity_sum
                pred_locs[choice] = get_distance(choice, pred_loc, edges)
            agent_choice = agent_loc
            if len(prey_locs) > 0 :
                agent_choice = get_full_information_choice(prey_locs, pred_locs, prey_distance, pred_distance,
                                                           agent_loc)
            prey_prob = update_prey_probs_by_agent(prey_prob, true_prey_loc, agent_choice)
            # print("prey probs after agent movement :", prey_prob)
            agent_loc = agent_choice
            if (true_prey_loc == agent_loc) :
                # print("Agent caught the prey")
                return True, False, step_count
            if (agent_loc == pred_loc) :
                # print("Predator caught the agent")
                return False, False, step_count
            true_prey_loc = move_prey(env, true_prey_loc)
            prey_prob = update_prey_probs_by_prey_movement(prey_prob, edges)
            pred_loc = easily_distracted_predator(env, pred_loc, agent_loc)
            step_count += 1
        return False, True, 0

    def partial_bonus_agent(self, env, utility) :
        agent_loc = env.agent_location
        true_prey_loc = env.prey_location
        prey_loc = None
        pred_loc = env.predator_location
        edges = env.edges
        step_count = 0
        prey_prob = init_prey_probs(agent_loc)

        while (True) :
            # print("Agent, prey and predator locations are ", agent_loc, true_prey_loc, pred_loc)
            if (agent_loc == pred_loc) :
                # print("Predator caught the agent u_partial")
                return False, False, step_count
            if (true_prey_loc == agent_loc) :
                # print("agent u_partial caught the prey")
                return True, False, step_count

            survey_node = choose_node_for_survey(prey_prob)
            if survey_node :
                prey_prob = update_prey_probs_by_survey(prey_prob, true_prey_loc, survey_node)

            agent_choices = edges[agent_loc]
            local_utility = {}

            for agent_choice in agent_choices :
                highest_probable_prey_loc = get_highest_prob(prey_prob)
                ag_pr_dist = 0
                for prey_neighbr in edges[highest_probable_prey_loc] + [highest_probable_prey_loc] :
                    if prey_prob[prey_neighbr] == 0 :
                        ag_pr_dist += get_distance(agent_choice, prey_neighbr, edges)
                    else :
                        ag_pr_dist += prey_prob[prey_neighbr] * get_distance(agent_choice, prey_neighbr, edges)
                ag_pr_dist = ag_pr_dist / len(edges[highest_probable_prey_loc]) + 1

                if agent_choice == pred_loc :
                    local_utility[agent_choice, pred_loc] = inf
                    continue
                elif (agent_choice == true_prey_loc) :
                    local_utility[agent_choice, pred_loc] = 0
                else :
                    pred_prob = compute_predator_probabilities(env, pred_loc, agent_choice)
                    lu = 0
                    for pred_choice in edges[pred_loc] :
                        ag_pred_choice_dist = get_distance(agent_choice, pred_choice, env.edges)
                        if (agent_choice == pred_choice) :
                            lu = lu + inf
                        else :
                            u_part = calculate_u_partial(utility, agent_choice, pred_choice, prey_prob)
                            lu = lu + pred_prob[pred_choice] * u_part
                        if ag_pred_choice_dist > 0 :
                            lu = lu / ag_pred_choice_dist
                        else :
                            lu = inf
                    local_utility[agent_choice, pred_loc] = lu

                    # ag_pr_dist = calculate_ag_prey_distance(env.edges,agent_choice,prey_prob)
                    local_utility[agent_choice, pred_loc] = local_utility[agent_choice, pred_loc] * ag_pr_dist

            local_utility = {k : v for k, v in sorted(local_utility.items(), key=lambda item : item[1])}

            if list(local_utility.keys())[0][0] != inf :
                agent_loc = list(local_utility.keys())[0][0]

            prey_prob = update_prey_probs_by_agent(prey_prob, true_prey_loc, agent_loc)

            step_count = step_count + 1
            if agent_loc == pred_loc:
                # print("Pred caught the Agent u_partial")
                return False, False, step_count
            if true_prey_loc == agent_loc :
                # print("Agent u_partial caught the prey")
                return True, False, step_count
            true_prey_loc = move_prey(env, true_prey_loc)
            prey_prob = update_prey_probs_by_prey_movement(prey_prob, edges)
            pred_loc = easily_distracted_predator(env, pred_loc, agent_loc)

    def bonus_agent_1(self, env) :
        agent_loc = env.agent_location
        true_prey_loc = env.prey_location
        prey_loc = None
        pred_loc = env.predator_location
        edges = env.edges
        step_count = 0
        prey_prob = init_prey_probs(agent_loc)
        
        indx = 0
        while(True):
          if(agent_loc == pred_loc):
            return False, False, step_count
          if(true_prey_loc == agent_loc):
            return True, False, step_count
         
          survey_node = choose_node_for_survey(prey_prob)
          if survey_node:
            prey_prob = update_prey_probs_by_survey(prey_prob, true_prey_loc, survey_node)
          
          agent_choices = edges[agent_loc]
          local_utility = {}

          for agent_choice in agent_choices:
            ag_prey_distance = calculate_ag_prey_distance(edges, agent_choice, prey_prob)
            agent_pred_distance = get_distance(agent_choice, pred_loc, env.edges)
            if agent_choice == pred_loc : 
              local_utility[agent_choice, pred_loc] = inf
              continue
            elif (agent_choice == true_prey_loc):
                local_utility[agent_choice, pred_loc] = 0
            else:
                local_utility[agent_choice, pred_loc] = predict_model_v_bonus_partial(agent_choice, pred_loc, prey_prob, ag_prey_distance, agent_pred_distance)
          
          local_utility = {k: v for k, v in sorted(local_utility.items(), key=lambda item: item[1])}
          agent_loc = list(local_utility.keys())[0][0]
          prey_prob = update_prey_probs_by_agent(prey_prob, true_prey_loc, agent_loc)
          step_count = step_count + 1
          if(agent_loc == pred_loc):
              return False, False, step_count
          if(true_prey_loc == agent_loc):
              return True, False, step_count
          true_prey_loc = move_prey(env, true_prey_loc)
          prey_prob = update_prey_probs_by_prey_movement(prey_prob, edges)
          pred_loc = easily_distracted_predator(env, pred_loc, agent_loc)

def main() :
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_1', action='store_true', default=False)
    parser.add_argument('--agent_2', action='store_true', default=False)
    parser.add_argument('--agent_3', action='store_true', default=False)
    parser.add_argument('--agent_4', action='store_true', default=False)
    parser.add_argument('--u_star_agent', action='store_true', default=False)
    parser.add_argument('--v_star_agent', action='store_true', default=False)
    parser.add_argument('--partial_bonus_agent', action='store_true', default=False)
    parser.add_argument('--u_partial', action='store_true', default=False)
    parser.add_argument('--bonus_agent_1', action='store_true', default=False)

    prey_certainity = {'agent_3' : 0,
                       'agent_4' : 0}

    args = parser.parse_args()
    env = The_Environment()

    agent = Agent()
    if args.agent_1 :
        print(agent.agent_1(env, 50))
    if args.agent_2 :
        print(agent.agent_2(env, 50))
    if args.agent_3 :
        print(agent.agent_3(env, 100, prey_certainity))
        print("Prey certainity is ", prey_certainity)
    if args.agent_4 :
        print(agent.agent_4(env, 100, prey_certainity))
        print("Prey certainity is ", prey_certainity)
    if args.u_star_agent :
        print(agent.u_star_agent(env, utility,policy))
    if args.v_star_agent :
        print(agent.v_star_agent(env, utility))
    if args.partial_bonus_agent :
        print(agent.partial_bonus_agent(env))
    if args.bonus_agent_1 :
        print(agent.bonus_agent_1(env, utility))
    if args.u_partial :
        print(agent.u_partial(env, utility))


if __name__ == '__main__' :
    main()