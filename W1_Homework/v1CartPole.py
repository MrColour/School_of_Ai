import gym
import numpy as np
import time #This is used to get an idea of what is happening, instead of it all happening all at once. Not necessary...

env = gym.make('CartPole-v0') #Make sure Pole is capitalized!
env.reset()

#How many steps do you want to run?
steps = 60
#Using 'print(env.action_space)' will show you the 'types' of actions.
#It may be type Discrete or Box. See: https://gym.openai.com/docs/ for more info.
#In our case 'print(env.action_space)' shows Discrete(2) meaning there are 2 possible actions to take
#Doing 'env.step(0)' produces one action, 'env.step(1)' produces the other action
#If it were Discrete(5) we would have aviable to us 'env.step(0)', 'env.step(1)', 'env.step(2)', 'env.step(3)', 'env.step(4)'

#Setting the initial action to something random. Remember since it is Discrete(2), it has to be a integer between 0 and 2. No floats!
action = np.random.randint(2)

#What is your guess? Try changing this value and see what happens!
guess = 1.5

for x in range(steps):
  #This shows the current environment. Using it once will show that window frozen in place without updates. So to see what happens we need continously render it.
  env.render()
  #The 'env.step(<numberhere>)' actually is a return function that preforms and!!! returns 'observation', 'reward', 'done', and 'info' in that order.
  lis = env.step(action)
  
  #lis[0] is the first item of 'env.step(action)'. It returns 'observation' which in this environment are
  #Cart Position, Cart Velocity, Pole Angle, Pole Velocity at Tip. So lis[0][0] returns the carts position.
  if (np.abs(np.abs(lis[0][0]) > 1.5):
    #Does an Automatic thingy that checks and then closes the Python window of 'env.render()' aka exits
    env.close()
    break
    
  if (lis[0][2] > np.deg2rad(guess)):
    #print("I should move to the right?")
    action = 1
    
  if (lis[0][2] > np.deg2rad(guess)):
    #print("I should move to the left?")
    action = 0
  
  #To slow things down a bit.
  time.sleep(.3)
  
#If nothing else closed env then close it.
env.close()
