from snake_game import snake_game
env = snake_game()
from keras.models import Sequential, load_model
from keras.layers import *
from keras.optimizers import RMSprop
import numpy as np
from collections import deque
import random
import cv2

inp = str(input('train from last? (y/n): '))
if inp == 'y':
    MODEL = str(input('pick a model: '))
    epsilon = 0
else:
    MODEL = None
    epsilon = 1
replay_memory = deque(maxlen=100_000)
MIN_REPLAY_SIZE = 1000
MINIBATCH_SIZE = 32
GAMMA = 0.99
UPDATE_COUNTER = 1000
EPISODES = 1000

def preprocess(img):
    return np.array(img).astype(np.uint8)
ENV_OBSERVATION_SPACE = env.observation_space

def dqn():
    if MODEL is not None:
        model = load_model(MODEL)
    else:
        model = Sequential()

        model.add(Conv2D(32,(3,3),strides=2,input_shape=ENV_OBSERVATION_SPACE))
        #model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Activation('relu'))

        model.add(Conv2D(64,(3,3),strides=1))
        #model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Activation('relu'))

        model.add(Flatten())

##        model.add(Dropout(0.1))
##        model.add(Dense(128,activation='relu'))
##        model.add(Dropout(0.1))
        model.add(Dense(64,activation='relu'))
        model.add(Dense(env.action_space))

        model.compile(loss='mse', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])
    return model

network = dqn()
tar_net = dqn()
tar_net.set_weights(network.get_weights())

def train(model, target_model, transition):
    if len(transition) < MIN_REPLAY_SIZE:
        return
    X = []
    y = []
    minibatch = random.sample(transition, MINIBATCH_SIZE)
    for state, a, r, is_terminal, next_state in minibatch:
        current_qs = model.predict(np.array(state).reshape(1, *ENV_OBSERVATION_SPACE)/255.0)[0]
        if is_terminal:
            new_q = r
        else:
            future_qs = target_model.predict(np.array(next_state).reshape(1, *ENV_OBSERVATION_SPACE)/255.0)[0]
            new_q = r + GAMMA * np.max(future_qs)
        current_qs[a] = new_q

        X.append(state)
        y.append(current_qs)
    model.fit(np.array(X).reshape(MINIBATCH_SIZE, *ENV_OBSERVATION_SPACE)/255.0, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0)

def main():
    counter = 0
    global epsilon
    INIT_EPSILON = epsilon
    FINAL_EPSILON = 0
    TARGET_STEPS = 5000
    last_record = int(input('last_record?'))
    for episode in range(EPISODES):
        current_state = preprocess(env.reset())
        done = False
        ep_reward = 0
        while not done:
            counter += 1
            if np.random.random() > epsilon:
                qs = network.predict(np.array(current_state).reshape(1, *ENV_OBSERVATION_SPACE)/255.0)[0]
                action = np.argmax(qs)
            else:
                action = np.random.randint(0, env.action_space)
            new_state, reward, done = env.step(action)
            new_state = preprocess(new_state)
            #reward = np.sign(reward)
            replay_memory.append([current_state, action, reward, done, new_state])
            train(network, tar_net, replay_memory)
            #env.render()
            #cv2.imshow('game_state',cv2.resize(current_state,(500,500)))
            #if cv2.waitKey(25) & 0xFF == ord('q'):
                #cv2.destroyAllWindows()
            current_state = new_state
            ep_reward += reward
            #if reward != 0:
                #print(reward)
            if counter >= UPDATE_COUNTER:
                
                counter = 0
                tar_net.set_weights(network.get_weights())
            if FINAL_EPSILON <= epsilon <= INIT_EPSILON:
                epsilon = (FINAL_EPSILON - INIT_EPSILON)/TARGET_STEPS * len(replay_memory) + INIT_EPSILON
        print('progress check: {} %'.format((episode/EPISODES)*100))
        if ep_reward > last_record:
            last_record = ep_reward
            network.save('models/snake-{}.model'.format(ep_reward))
            print('new_best_score is:{}'.format(ep_reward))
        network.save('snakeai-v5.model')



if __name__ == '__main__':
    main()
