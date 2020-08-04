import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

LR = 1e-3
env = gym.make('CartPole-v0')
env._max_episode_steps = 500
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 10000

def some_random_games_first():
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break

# some_random_games_first()

def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(10000):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            action = random.randrange(0,2)
            observation, reward, done, info = env.step(action)
            
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            
            prev_observation = observation
            score += reward
            if done:
                break
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]
                training_data.append([data[0], output])
        env.reset()
        scores.append(score)
    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    print('Avrage accepted score:', mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data

# initial_population()

def neural_network_model(input_size):
    network = input_data(shape=[None, input_size, 1], name='input')
    print(network)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]))
    
    model.fit({'input':X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='epenaistuff')

    return model

training_data = initial_population()
model = train_model(training_data)

scores = []
choices = []
for each_game in range(30):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        env.render()
        if len(prev_obs) == 0:
            action = random.randrange(0, 2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])
        choices.append(action)

        new_observation, reward, done, info = env.step(action)
        
        if action == 1:
            actionOH = [0,1]
        elif action == 0:
            actionOH = [1,0]
        if len(prev_obs) > 0:
            game_memory.append([prev_obs, actionOH])

        prev_obs = new_observation
        score += reward
        if done:
            break
    scores.append(score)
    model = train_model(game_memory, model)
    # X = np.array([i[0] for i in game_memory]).reshape(-1, len(game_memory[0][0]), 1)
    # y = [i[1] for i in game_memory]
    # model.fit({'input':X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='epenaistuff')
    game_memory = []
    prev_obs = []

print('Top score', max(scores))
print('All scores', scores)
print('Choice 1: {}, Choice 2: {}'.format(choices.count(1)/len(choices), choices.count(0)/len(choices)))
