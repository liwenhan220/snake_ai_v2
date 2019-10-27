from snake_game import snake_game
env = snake_game()
from keras.models import load_model, Sequential
from keras.layers import *
from keras.optimizers import RMSprop
import numpy as np
import cv2


path = str(input())
if path == '':
    MODEL = 'snakeai-v5.model'
else:
    MODEL = 'models/{}'.format(path)
EPISODES = 10000

network = load_model(MODEL)
def main():

    for episode in range(EPISODES):
        current_state = env.reset()
        done = False
        ep_reward = 0
        while not done:

            qs = network.predict(np.array(current_state).reshape(1, *env.observation_space)/255.0)[0]
            action = np.argmax(qs)
            new_state, reward, done = env.step(action)
            env.render()
            current_state = new_state
            ep_reward += reward

if __name__ == '__main__':
    main()
