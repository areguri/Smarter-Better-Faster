# READDDD PICKLE FILES
import os
import sys
import pandas as pd

import pickle
sys.path.append(os.path.abspath(__file__))

from environment import The_Environment
from agent import Agent


with open('edges.pickle', 'rb') as f:
  edge_pick = pickle.load(f)

with open('utility.pickle', 'rb') as f:
  utility_pick = pickle.load(f)

with open('policy.pickle', 'rb') as f:
  policy_pick = pickle.load(f)

with open('reward.pickle', 'rb') as f:
  reward_pick = pickle.load(f)

env = The_Environment()
env.edges = edge_pick
utility = utility_pick
reward = reward_pick
policy = policy_pick

pred = pd.read_csv('predictions_v_star.csv', index_col=0)
v_dict = {}
for i in range(pred.shape[0]):
  agent_loc = pred.iloc[i]['Agent_loc']
  prey_loc =  pred.iloc[i]['Prey_loc']
  pred_loc =  pred.iloc[i]['Pred_loc']
  v_dict[agent_loc, prey_loc, pred_loc] = pred.iloc[i]['predicted_v']

agent = Agent() 
wins = {"agent_1":0, "agent_2":0, "u_star_agent":0, "v_star_agent":0, "agent_3":0, "agent_4":0, "u_partial":0, "v_partial":0, "partial_bonus_agent":0, "bonus_agent_1":0}
timeouts = {"agent_1":0, "agent_2":0, "u_star_agent":0, "v_star_agent":0, "agent_3":0, "agent_4":0, "u_partial":0, "v_partial":0, "partial_bonus_agent":0, "bonus_agent_1":0}
step_counts = {"agent_1":0, "agent_2":0, "u_star_agent":0, "v_star_agent":0, "agent_3":0, "agent_4":0, "u_partial":0,"v_partial":0, "partial_bonus_agent":0, "bonus_agent_1":0}

print("Prey, predator and agent locations are ", env.prey_location, env.predator_location, env.agent_location)
for i in range(3000):
  env.generate_ppa()
  success, timeout, step_count, _, _, _ = agent.u_star_agent(env, utility, policy)
  if success:
    wins["u_star_agent"] += 1
  timeouts["u_star_agent"] += timeout
  step_counts["u_star_agent"] += step_count

  success, timeout, step_count, _, _, _ = agent.v_star_agent(env, v_dict)
  if success:
    wins["v_star_agent"] += 1
  timeouts["v_star_agent"] += timeout
  step_counts["v_star_agent"] += step_count

  success, timeout, step_count, _, _, _ = agent.agent_1(env, 100)
  if success:
    wins["agent_1"] += 1
  timeouts["agent_1"] += timeout
  step_counts["agent_1"] += step_count

  success, timeout, step_count, _, _, _ = agent.agent_2(env, 100)
  if success:
    wins["agent_2"] += 1
  timeouts["agent_2"] += timeout
  step_counts["agent_2"] += step_count

  success, timeout, step_count, _ = agent.u_partial(env, utility)
  if success:
    wins["u_partial"] += 1
  timeouts["u_partial"] += timeout
  step_counts["u_partial"] += step_count

  success, timeout, _ = agent.v_partial(env)
  if success:
    wins["v_partial"] += 1
  timeouts["v_partial"] += timeout
  step_counts["v_partial"] += step_count

  success, timeout, step_count = agent.agent_3(env, 100)
  if success:
    wins["agent_3"] += 1
  timeouts["agent_3"] += timeout
  step_counts["agent_3"] += step_count

  success, timeout, step_count = agent.agent_4(env, 100)
  if success:
    wins["agent_4"] += 1
  timeouts["agent_4"] += timeout
  step_counts["agent_4"] += step_count

  success, timeout, step_count = agent.partial_bonus_agent(env, utility)
  if success:
    wins["partial_bonus_agent"] += 1
  timeouts["partial_bonus_agent"] += timeout
  step_counts["partial_bonus_agent"] += step_count

  success, timeout, step_count = agent.bonus_agent_1(env)
  if success:
    wins["bonus_agent_1"] += 1
  timeouts["bonus_agent_1"] += timeout
  step_counts["bonus_agent_1"] += step_count

for i in step_counts:
  step_counts[i] = step_counts[i]/3000

for k in wins.keys():
  worker = k
  print("Analysis of ", worker)
  print("Success rate: ", (wins[k]/3000)*100)
  print("Average stepcount :", step_counts[k])
  print("****************************************************************************")