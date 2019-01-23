import gym
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
import random

env = gym.make("CartPole-v1")
observation = env.reset()
game_memory = []
accepted_scores = []

def random_games():
	for _ in range(100000):
		env.reset()
		done = False
		observation_data = []
		score = 1
		prev_ob = []
		while done is not True:

			action = env.action_space.sample()
			observation, reward, done, info = env.step(action)
			
			
			if len(prev_ob) > 0 :
				data = [prev_ob[0],prev_ob[1],prev_ob[2], prev_ob[3], action]
				observation_data.append(data)
			prev_ob = observation
			score += reward

		if score > 100:
			accepted_scores.append(score)
			for element in observation_data:
				element.append(score)
				game_memory.append(element)
	return len(accepted_scores), game_memory


def train_model(game_memory):
	df = pd.DataFrame(game_memory, columns = ['ob0','ob1','ob2','ob3','action', 'score'])
	if len(game_memory) > 20000:
		df = df.tail(20000).sort_values('score')
		game_memory = df.values.tolist()

	df1 = np.array(df.drop(['action','score'], axis = 1))
	y = np.array(df['action'])
	mlp = MLPRegressor(hidden_layer_sizes=(50,100, 200,100,50), verbose = False, max_iter=20000,tol=0.00001)#, n_iter_no_change = 1000, solver = 'adam')
	mlp.fit(df1,y)

	

	return mlp, game_memory


def test_model(model, iterations, render, score_increaser, game_memory):
	observation = [0,0,0,0]
	accepted_scores = []
	for _ in range(iterations):
		env.reset()
		done = False
		observation_data = []
		score = 1
		prev_ob = []
		score_keeper = []
		while done is not True:
			if render == True:
				env.render()
			df4 = pd.DataFrame.from_dict({'obs1': [observation[0]],'obs2': [observation[1]],'obs3': [observation[2]],'obs4':[observation[3]]})
			data = np.array(df4)
			action = int(np.round(model.predict(data)))

			if action > 1:
				action = 1
			if action < 0:
				action = 0
			observation, reward, done, info = env.step(action)

			if len(prev_ob) > 0:
				data = [prev_ob[0],prev_ob[1],prev_ob[2], prev_ob[3], action]
				observation_data.append(data)
			prev_ob = observation
			score += reward

		score_keeper.append(score)
		required_score = 180 + score_increaser
		if score > required_score:
			accepted_scores.append(score)
			for element in observation_data:
				game_memory.append(element)
		if render == True:
			env.close()
	if len(accepted_scores) > 0:
		print('The average accepted score was {}, required score was {}'.format(np.array(accepted_scores).mean(), required_score))
	else: 
		print('The were no accepted scores that met the threshold of {}'.format(required_score))
	print('The average score was {}'.format(np.array(score_keeper).mean()))
	return len(accepted_scores), game_memory

def train(epochs, iterations):
	old_num_good_scores, game_memory = random_games()
	print('game memory length is {}'.format(len(game_memory)))
	i = 1
	for _ in range(epochs):
		score_increaser = 0
		model, game_memory = train_model(game_memory)
		num_good_scores, game_memory = test_model(model, iterations, False, score_increaser, game_memory)

		print('finished the {}ith epoch'.format(i))

		print('Number of good retunred scores {}, goal sucess rate of {}'.format(num_good_scores, num_good_scores/iterations))
		print('The length of the game memory is {}'.format(len(game_memory)))
		if num_good_scores/iterations == 0.0:
			break
		i += 1
	test_model(model, 50, True, 15, game_memory)


train(epochs=10 ,iterations=100)



