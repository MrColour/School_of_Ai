import gym
import numpy as np
import time

env = gym.make('CartPole-v0')

settings = {
    'maxdis': 1.5,
    #'steps' determine the max reward and how long each session is. 'wait' is just to slow things down and see what is happening.
    'steps': 500, 'wait': .01,
    #Below are the 'weights'
    'wfCartPos': 0, 'wfCartVel': -.09, 'wfPoleAngle': .7, 'wfPoleVelTip': -.054,
    'attempts': 100
    }

s = settings

def calAction(obs):
    '''Calculates an Action'''
    if (s['wfCartPos']*obs[0] + s['wfCartVel']*obs[1] + s['wfPoleAngle']*obs[2] + s['wfPoleVelTip']*obs[3] > 0):
        return 1
    else:
        return 0 

bestreward = 0
bestweights = [s['wfCartPos'], s['wfCartVel'], s['wfPoleAngle'], s['wfPoleVelTip']]

for i in range(s['attempts']):
    obs = env.reset()
    totalreward = 0

    for i in range(s['steps']):
        env.render()
        #Calculates an action based on the observations 'obs'. It is either 1 or 0. See v1CartPole for reason.
        action = calAction(obs)
        obs, reward, done, info = env.step(action)
        totalreward =  totalreward + reward
        #One of the conditions of done is being able to get rewarded 200 times.
        if (done and not(totalreward > 199)) or np.abs(obs[0]) > s['maxdis']:
            break

        time.sleep(s['wait'])
    
    if (totalreward >= bestreward):
        bestreward = totalreward
        print('New Best reward of:%d' % bestreward)
        bestweights = [s['wfCartPos'], s['wfCartVel'], s['wfPoleAngle'], s['wfPoleVelTip']]
        
    #Here is where you would add code to change your weights. Be choose randomly, Hillclimbing tries, or other.
    #Why haven't you added code then!! Lazy bum. 
    
env.close()
