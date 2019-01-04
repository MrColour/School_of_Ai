import gym
import numpy as np
import v020HwFrozenLakeKC
import v020HwAlg as Alg

env = gym.make('FrozenLakeKC-v0')
env.reset()

#Action to number dictionary, Anum
#Spaces are for formatting. So each has 5 characters. Too lazy to figure out proper way =[
actnum = {0:'Left ', 1:'Down ', 2:'Right', 3:'Up   ', -1:'None '}
rows = env.size('r')
cols = env.size('c')


random_policy = np.ones([env.nS, env.nA]) / env.nA
#for each state (env.nS) each action (env.nA) will have equal probability of being picked (dividing by total number of actions)
#So each state has four actions which will have 25% chance of being picked when in that state

v = Alg.policy_eval(random_policy, env)
policyAct = [0 for s in range(env.nS)]

#For each position (state) find the best action
for s in range(env.nS):
    policyAct[s] = Alg.action_bState(s, v, env)

print('\nThe following shows the value of being in such position:\n')
for r in range(rows):
    text = ''
    for c in range(cols):
        addon = format(v[c + r*cols], '.4f') + ' '
        text = text + addon
    print(text)

print('\nThe follwing shows which actions to take based on the being in that position:\n')
for r in range(rows):
    text = ''
    for c in range(cols):
        addon = actnum[policyAct[c + r*cols]] + ' '
        text = text + addon
    print(text)

state = 0
playing = True
m = 0
env.render(m)
#How many moves do you want the agent to get?
while (playing and m < 100):
    obs, reward, done, info =  env.step(policyAct[state])

    if (reward == 1):
        playing = False
    m = m + 1
    state = obs
    env.render(m)

env.close()