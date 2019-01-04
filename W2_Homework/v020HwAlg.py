import numpy as np

def policy_eval(policy, env, discount_factor=1.0, closeenough=0.00001):
#closeenough is theta here. Since posValue 'converges', closeenough tells us when to stop process of converging

    posValue = [0 for s in range(env.nS)]
    #Initializing the value of every state to zero (env.nS returns the 'number of States').
    #In this case every state is a position on the grid. Thus I call it the position's value, posValue.

    while True:
        valdif = 0
        #valdif is a bit confusing here since it is zero but later in the code it will be the max of two things: itself or difference of expected value and recorded value.
        #If valdif is 0 that means the value of that position has converged. Since it is not likely to be zero in the first run, valdif will be difference of expected value and recorded value.
        #So it is zero here only for the convenience of not doing a first run check and initialization. 

        for s in range(env.nS):
            v = 0
            #v will be the variable that we will be using to check the expected value. We will then record this to posValue and
            #posValue eventually it will converge to the actual value of each position.

            for a, action_prob in enumerate(policy[s]):
                #for each action our policy allows

                for prob, next_state, reward, done in env.P[s][a]:
                    #We list out [prob, next_state, reward, done] since we are saying for each tuple of 4 in env.P[s][a]. Python stuff?

                    v += action_prob * prob * (reward + discount_factor * posValue[next_state])
                    #So this is saying the expected reward for taking action a is:
                    #The probability of taking that action * the probability of sucessfully going where the action says * (the immediate reward + the expected future reward (discounted))
                    #Quite a mouthful!

            valdif = max(valdif, np.abs(v-posValue[s]))
            posValue[s] = v
            #Notice here that since we are in the loop of s in range(env.nS) we are going to check all the states
            #Meaning that the 'last' state or the one before finishing the game taking the winning action the game HAS to be greater than zero!
            #Meaning! that valdif will most likely be greater than closeenough

        print('We are currently ' + str(valdif) + " close!")             
        if (valdif < closeenough):
            print('\nWe have converged, congratz! Insert celebration here!\n')
            break
    
    return np.array(posValue)

def action_bState(state, v, env, discount_factor = 1.0):
    '''Returns the action that would lead to the best State as predicted by our value 'v' '''
    #In some cases this might not be the best of ideas?
    actValue = [-1 for a in range(env.nA)]

    for a in range(env.nA):
        #Try out all four actions and see the value of 'next_state'
        for prob, next_state, reward, done in env.P[state][a]:
            actValue[a] += prob * (reward + discount_factor * v[next_state])
            #print('Doing action ' + str(a) + ' moves you to ' + str(next_state) + ' with a value of ' + str(v[next_state]))
    
    bAv = -1
    #Null value, -1 action isn't a choice, meaning undecidable
    bAct = -1
    for i in range(len(actValue)):
        #You can check iF there are two actValues with the same value here to randomly pick one.
        if (actValue[i] > bAv):
            bAct = i
            bAv = actValue[i]
    
    return bAct

