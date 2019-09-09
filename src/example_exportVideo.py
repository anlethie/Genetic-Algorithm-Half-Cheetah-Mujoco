import gym
from actor import Actor
from gym import wrappers

env = gym.make('Ant-v2')
env = wrappers.Monitor(env, '/home/an/cs169-render/Ant-v2-example-2', force=True)
for i_episode in range(10):
    actor = Actor(env.observation_space, env.action_space)
    observation = env.reset()
    for t in range(20000):
        #rgb_array = env.render(mode='rgb_array')
        # print(observation)
        action = actor.react_to(observation)
        print(action)
        # print(action)
        observation, reward, done, info = env.step(action)
        #print(observation)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            print("Total reward: ", str(reward))
            break

env.close()