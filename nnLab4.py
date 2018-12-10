from laberint_game import LaberintGame
from random import randint
import numpy as np
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean
from collections import Counter

class LaberintNN:
    def __init__(self, initial_games = 1000, test_games = 1, goal_steps = 200, lr = 1e-2, filename = 'Lab1_nn.tflearn'):
        self.initial_games = initial_games
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.lr = lr
        self.filename = filename
        self.vectors_and_keys = [
                [[0, -1], 0], # Izquierda
                [[1, 0], 1],  # abajo
                [[0, 1], 2],  # derecha
                [[-1, 0], 3]  # arriba
                ]
    
    # para crear el dataset que va ayudar a entrenar la red neuronal
    def initial_population(self):
        training_data = []
        for _ in range(self.initial_games):
            game = LaberintGame()
            _, _, board, x, y, lastAction = game.start()
            prev_observation = self.generate_observation(x, y, board, lastAction)
            old_x, old_y = x, y 
            for _ in range(self.goal_steps):
                action, game_action = self.generate_action(x, y, lastAction)
                done, _, board, x, y, lastAction  = game.step(game_action)
                if done:
                    if board[x][y] == "t" :
                        training_data.append([self.add_action_to_observation(prev_observation, action), 0])    
                    else:
                        training_data.append([self.add_action_to_observation(prev_observation, action), 1])     
                    break 
                else:
                    if (x == old_x) & (y == old_y) :
                        training_data.append([self.add_action_to_observation(prev_observation, action), 0]) 
                    else:
                        training_data.append([self.add_action_to_observation(prev_observation, action), 1]) 
                # con estos valores, vamos a tener un vector y unicamente compuesto de 0 y 1 para simplificar el modelo 
                prev_observation = self.generate_observation(x, y, board, lastAction)
                old_x, old_y = x, y                    
        print(len(training_data))
        return training_data

    # generar la accion al azar
    def generate_action(self, x, y, lastAction):
        action = randint(0,3) 
        return action, self.get_game_action(action, lastAction)

    # generar el game_action usando la accion pasada y la accion generada al azar
    def get_game_action(self, action, lastAction):
        a_direction = self.get_a_direction_vector(lastAction)
        new_direction = a_direction
        if action == 0:
            new_direction = self.turn_vector_to_the_left(a_direction)
        elif action == 1:
            new_direction = self.turn_vector_to_the_back(a_direction)
        elif action == 2:
            new_direction = self.turn_vector_to_the_right(a_direction)
        for pair in self.vectors_and_keys:
            if pair[0] == new_direction.tolist():
                game_action = pair[1]
        return game_action

    # generar le vector que dice si una direccion esta libre o bloqueada
    def generate_observation(self, x, y, board, lastAction):
        a_direction = self.get_a_direction_vector(lastAction)
        barrier_left = self.is_direction_blocked(x, y, board, self.turn_vector_to_the_left(a_direction))
        barrier_front = self.is_direction_blocked(x, y, board, a_direction)
        barrier_right = self.is_direction_blocked(x, y, board, self.turn_vector_to_the_right(a_direction))
        barrier_back = self.is_direction_blocked(x, y, board, self.turn_vector_to_the_back(a_direction))        
        return np.array([int(barrier_left), int(barrier_back), int(barrier_right), int(barrier_front)])

    # agregar el vector action a al vector observation
    def add_action_to_observation(self, observation, action):
        return np.append(self.array_action(action), observation)
        #return np.append([action], observation)

    # obtener la direction del punto A gracias a la ultima accion
    def get_a_direction_vector(self, lastAction):
        if lastAction == 'LEFT':
            return np.array([0, -1])
        elif lastAction == 'DOWN':
            return np.array([1, 0])
        elif lastAction == 'RIGHT':
            return np.array([0, 1])
        else: #lastAction == 'UP':
            return np.array([-1, 0])

    # saber si una direccion esta bloqueada
    def is_direction_blocked(self, x, y, board, direction):
        point = np.array([x, y]) + np.array(direction)
        [i, j] = point        
        if board[i][j] == 't':
            return 0              # decidi poner la misma valor para "t" y "w" para tener un vector X
        elif board[i][j] == 'w':  # compuesto unicamente de 0 y 1 
            return 0
        else:
            return 1

    def turn_vector_to_the_left(self, vector):
        return np.array([-vector[1], vector[0]])

    def turn_vector_to_the_right(self, vector):
        return np.array([vector[1], -vector[0]])
    
    def turn_vector_to_the_back(self, vector):
        return np.array([-vector[0], -vector[1]])
        
    # escribi la accion asi para que la red neuronal no se equivoca con los nombres 0,1,2,3
    def array_action(self, action):        
        if action == 0:
            return np.array([1, 0, 0, 0])
        elif action == 1:
            return np.array([0, 1, 0, 0])
        elif action == 2:
            return np.array([0, 0, 1, 0])
        else: #actions == 3:
            return np.array([0, 0, 0, 1])
    
    # armar el modelo
    def model(self):
        network = input_data(shape=[None, 8, 1], name='input')
        network = fully_connected(network, 128, activation='leaky_relu')  #leaky_relu para asegurarse de no apagar una neurona     
        network = fully_connected(network, 1, activation='linear')
        network = regression(network, optimizer='adam', learning_rate=self.lr, loss='mean_square', name='target')
        model = tflearn.DNN(network, tensorboard_dir='log')
        return model
        
    # entrenar el modela    
    def train_model(self, training_data, model):
        X = np.array([i[0] for i in training_data]).reshape(-1, 8, 1)
        y = np.array([i[1] for i in training_data]).reshape(-1, 1)        
        model.fit(X,y, n_epoch = 1, shuffle = True, run_id = self.filename) 
        model.save(self.filename)
        return model
        
    # test del modelo
    def test_model(self, model):
        steps_arr = []
        for _ in range(self.test_games):
            steps = 0
            game_memory = []
            game = LaberintGame()
            _, _, board, x, y, lastAction = game.start()
            prev_observation = self.generate_observation(x, y, board, lastAction)
            for _ in range(self.goal_steps):
                predictions = []
                for action in range(0, 4):
                   predictions.append(model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, 8, 1)))
                if predictions[0] > 0.5: #aca se usa la regla de la mano izquierda 
                    action = 0           #ir a la izquierda
                else:
                    if predictions[3] > 0.5:
                        action = 3       #ir adelante
                    else:
                        if predictions[2] > 0.5:
                            action = 2   #ir a la derecha
                        else:
                            action = 1   # volver atras
#                print(predictions)            
#                print(action)
#                print(predictions[action])
                game_action = self.get_game_action(action, lastAction)
#                print(game_action)
                done, _, board, x, y, lastAction = game.step(game_action)
                game_memory.append([prev_observation, action])
                if done:
                    break
                else:
                    prev_observation = self.generate_observation(x, y, board, lastAction)
                    steps += 1
            steps_arr.append(steps)
        print('Average steps:',mean(steps_arr))
        print(Counter(steps_arr))

    # para jugar al laberinto
    def visualise_game(self, model):
        game = LaberintGame(gui = True)
        _, _, board, x, y, lastAction = game.start()
        prev_observation = self.generate_observation(x, y, board, lastAction)
        for _ in range(self.goal_steps):
            predictions = []
            for action in range(0, 4):
               predictions.append(model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, 8, 1)))
            if predictions[0] > 0.5:
                action = 0
            else:
                if predictions[3] > 0.5:
                    action = 3
                else:
                    if predictions[2] > 0.5:
                        action = 2
                    else:
                        action = 1
            game_action = self.get_game_action(action, lastAction)
            done, _, board, x, y, lastAction  = game.step(game_action)
            if done:
                break
            else:
                prev_observation = self.generate_observation(x, y, board, lastAction)

    def train(self):
        training_data = self.initial_population()
        nn_model = self.model()
        nn_model = self.train_model(training_data, nn_model)
        self.test_model(nn_model)

    def visualise(self):
        nn_model = self.model()
        nn_model.load(self.filename)
        self.visualise_game(nn_model)

    def test(self):
        nn_model = self.model()
        nn_model.load(self.filename)
        self.test_model(nn_model)

if __name__ == "__main__":
    #LaberintNN().train()
    LaberintNN().visualise()
    #LaberintNN().test()


