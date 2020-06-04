#Import the dependencies.
from collections import deque                          # Store for the stacking the frames and the average score of the last 10 consecustive episodes (to see improvement)
import matplotlib.pyplot as plt                        # To plot the graph of average episodes over time
from keras import backend as K 

from tqdm import tqdm                                  # To show the loading bar

import random                                          # Used for the Epsilon-Greedy method
import numpy as np                                     # Used for matrix calculations and other calculations.
import os                                              # To save the File
import retro                                           # To load the game.

#Dependecies for the Q-Deep Networks
from keras.models import Sequential                    # Imports the Sequential Model which allows to have multiple layers.
from keras.layers import Dense, Conv2D, Flatten        # Imports Convolutional, Flatten and Fully Connected layers
from keras.optimizers import RMSprop                   # Imports optimizers RMSprop


#The hyper-parameteres are based on this article (https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)

EPISODES = 50                                           # Maximum number of episodes
MAX_STEPS = 50000                                       # Maximum number of steps an agent can take in one episode
MEMORY_SIZE = 1000000                                   # Size of the memory that stores agent's experiences
REPLAY_SIZE = 50000                                     # A random policy is run for this number of frames before learning stats in order to populate the memory.
TARGET_UPDATE_FREQ = 10000                              # The frequency at which the target network is updated (every 10000 iterations)
BATCH_SIZE = 64                                         # Number of training cases   
HISTORY_SIZE = 4                                        # The number of most recent frames experienced by the agent that are given as input to the Q network.
INPUT_SHAPE = (105, 80, 4)                              # The Input shape of the frame that is going to be inputted into the Q-deep network.
FRAME_SIZE = (105,80)
#Parameters
ENV_NAME = 'SpaceInvaders-Atari2600'
NORMALIZE_VALUES = 255

class DNQAgent:
    def __init__(self, env):        
        self.input_shape = INPUT_SHAPE                                                            # Dimensions of the input image
        self.action_size = env.action_space.n                                                     # The size of possible actions made by an agent
        self.possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())      # Creates an array of all possible actions
        self.memory = deque(maxlen = MEMORY_SIZE)                                                 # Samples all the experiences that are then used for training the agent
        #Hyper parameters
        self.gamma = 0.99                                                                         # Discount Factor for Q Deep learning
        #EPSILON GREEDY 
        self.epsilon_greedy = 0                                                                   # Probability used for exploration (Exploitation vs Exploration dilemma)
        self.epsilon_min = 0.1                                                                    # The final (minimum) value for the epsilon
        self.epslon_start = 1                                                                     # The initial probability for exploration
        self.decay_rate = 0.00002                                                                 # The decay at which the probability for exploration will decrease
        #Pre-processing and stacking the frames
        self.stack = deque([np.zeros(FRAME_SIZE, dtype=np.uint8) for i in range(HISTORY_SIZE)], maxlen= HISTORY_SIZE)   # Stacks the 4 frames

    def atari_network(self):
        '''
            The Q-Network was based on the DeepMind article on Nature (https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)
        '''
        model = Sequential()                                        
        model.add(Conv2D(32,                              #First Convolutional layer - 32 of 8x8 filters
                         8,
                         strides=(4, 4),
                         padding="valid",
                         activation="relu",
                         input_shape=self.input_shape
                         ))
        model.add(Conv2D(64,                              #Second Convolutional layer  - 64 of 4x4 filters                     
                         4,
                         strides=(2, 2),
                         padding="valid",
                         activation="relu",
                         input_shape=self.input_shape
                         ))
        model.add(Conv2D(64,                              #Third Convolutional layer - 64 of 3x3 filters
                         3,
                         strides=(1, 1),
                         padding="valid",
                         activation="relu",
                         input_shape=self.input_shape
                        ))
        model.add(Flatten())                              #Flattens the data
        model.add(Dense(512 , activation="relu"))         #512 hidden neurons
        model.add(Dense(self.action_size))                #The Output layer
        
        #Uses the RMprop optimizer with the same parameters used in the article.
        model.compile(loss="mse",optimizer=RMSprop(lr=0.00025,rho=0.95,epsilon=0.01),metrics=["accuracy"])  
        return model
    
    #Append agent's observation to the memory.
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) 

    #It is used while the agent is observing the envrionment.
    #Returns a random action performed by an action. 
    def random(self):                                                                       
        action = np.random.choice(self.action_size)                                                 
        return self.possible_actions[action]


    #Returns a normalized state value
    def normalize(self, state):
        state = state / NORMALIZE_VALUES
        normalized_state = np.expand_dims(state, axis=0)
        return normalized_state

    def greedy(self, state, action_model, decay_step):
        # More efficient way of delaying the probability of epsilon greeedy approach. 
        # It has to explore roughly 1,000,000 frames before the value of the epsilon goes to 0.1 (its final value)
        self.epsilon_greedy = self.epsilon_min + ((self.epslon_start  - self.epsilon_min) * np.exp(-self.decay_rate * decay_step))
        if np.random.rand() <= self.epsilon_greedy:            
            action = np.random.choice(self.action_size)
            return self.possible_actions[action]
        state = self.normalize(state)                         # Normalizes the state.             
        choice = action_model.predict(state)                  # Uses the model to predict the q values and chooses the one that has the maximum value. 
        index = np.argmax(choice)
        return self.possible_actions[index]

    #Replay Experience that 
    def replay(self, action_model):
        minibatch = random.sample(self.memory, BATCH_SIZE)                                                     
        # Assigning the values from minibatch to numpy arrays
        states_minibatch = np.array([row[0] for row in minibatch]) / NORMALIZE_VALUES
        actions_minibatch = np.array([row[1] for row in minibatch]) 
        rewards_mnibatch = np.array([row[2] for row in minibatch])
        next_states_minibatch = np.array([row[3] for row in minibatch]) / NORMALIZE_VALUES
        done_minibatch = np.array([row[4] for row in minibatch])
        # Predictions for next states
        predicted = action_model.predict(next_states_minibatch)
        # Create an empty targets list to hold our q values
        target_minibatch = []
        for it in range(BATCH_SIZE):
            done = done_minibatch[it]
            # If game is terminated, we use the immediate reward
            if done:
                q_value = rewards_mnibatch[it]
            else:
                # If game is not terminated, we use q learning update rule to update our q value
                q_value = rewards_mnibatch[it] + self.gamma * np.max(predicted[it])
            # Assigns the q_value to the target array
            target = actions_minibatch[it] * q_value
            target_minibatch.append(target)

        targets_minibatch = np.array(target_minibatch)
        action_model.fit(states_minibatch, targets_minibatch, epochs=1, verbose=0)

    #Resizes the fame to (105,80) and then greyscales it
    def preprocess_frame(self, frame):
        frame = frame[::2, ::2]  
        return np.mean(frame, axis=2).astype(np.uint8)
        
    #It preprocesses a frame and then appends it to the stack.
    def add_to_stack(self,state,reset=False):
        processed_frame = self.preprocess_frame(state)
        if reset:
            self.stack = deque([np.zeros(FRAME_SIZE, dtype=np.int) for i in range(HISTORY_SIZE)], maxlen=HISTORY_SIZE)     # Reset the stack  
            # Copy the same frame four times because we are in new episode
            for i in range(HISTORY_SIZE):
                self.stack.append(processed_frame)
        else:
            self.stack.append(processed_frame)                # Just add a frame to the stack.
        stacked_state = np.stack(self.stack, axis=2)          # Stacks the frames.
        return stacked_state

#Draws a graph based on the average score per episode.
def graph(data):
    plt.plot(data,label="precision")
    plt.xlabel("Training epochs")
    plt.ylabel("Average Score per episode : ")
    plt.legend()
    plt.show()

    
if __name__ == "__main__":
    # Creates the envrionment for the game
    env = retro.make(ENV_NAME)         
    agent = DNQAgent(env)     
    # Creating the conv network
    action_model = agent.atari_network()                              
    # Stores all the rewards throughout the episodes to plot a graph
    all_episodes_rewards = []   
    # Decay_step to adjusts the epsilon for the epsilon greedy approach  
    decay_step  = 0                                                    

    print("Observing the Envrionment")
    
    # Observng the Envrionment ()
    for observetime in range(EPISODES):
        # Step 1: Reset the envrionment and the frame stack.
        if observetime == 0:
            state = env.reset()
            state  = agent.add_to_stack(state, True)
         # Step 2: Select a random action a_t
        action = agent.random()
        # Step 3: Execute this random action a_t in emulator and observe reward r_t and get state s_t+1
        next_state, reward, done, _ = env.step(action)
        if done:
            # Because its been terminated, it creates a blank next state and is then appended to the stack.
            next_state = np.zeros((210,160,3), dtype=np.int)
            next_state = agent.add_to_stack(next_state)
            # Step 4: Stores the transition <s_t,a_t,r_t+1,s_t+1> in memory D
            agent.remember(state, action, reward, next_state, done)
            # Goes back to Step 1: Reset the envrionment and the frame stack.
            state = env.reset()
            state = agent.add_to_stack(state, True)
        else:
            next_state = agent.add_to_stack(next_state)
            # Step 4: Stores the transition <s_t,a_t,r_t+1,s_t+1> in memory D
            agent.remember(state, action, reward, next_state, done)
            state = next_state
      
    print("Finished")
    
    for e in tqdm(range(EPISODES)): 
        #Step 1: Reset the envrionment and the frame stack.
        state = env.reset() 
        state  = agent.add_to_stack(state, True)
        done = False
        total_reward = 0
        episode_rewards = []
        for steps in range(MAX_STEPS):
            decay_step +=1
            # Step 2: Select an action a_t basedd on the epsilon greedy aporoach.
            action = agent.greedy(state, action_model,decay_step)
            # Step 3: Execute action a_t in emulator and observe reward r_t and get state s_t+1
            next_state, reward, done, _ = env.step(action)
            episode_rewards.append(reward)
            if done: 
                # Because its been terminated, it creates a blank next state and is then appended to the stack.
                next_state = np.zeros((210,160,3), dtype=np.int)
                next_state = agent.add_to_stack(next_state)
                # Gets the total reward of the episode
                total_reward = np.sum(episode_rewards)
                all_episodes_rewards.append(total_reward)
                # Step 4: Stores the transition <s_t,a_t,r_t+1,s_t+1> in memory D
                agent.remember(state, action, reward, next_state, done)    
                # Print the information 
                print('Episode: {}'.format(e),'Total reward: {}'.format(total_reward),'Explore P: {:.4f}'.format(agent.epsilon_greedy))
                # Start a new episode (because it leaves the nested loop)
                break
            # If not termianted the, adds the next state s_t+1 to the stack of frames (where the oldest one is deleted because its deque)
            next_state = agent.add_to_stack(next_state)
            #Step 4: Stores the transition <s_t,a_t,r_t+1,s_t+1> in memory D
            agent.remember(state, action, reward, next_state, done)
            #The state s_t becomes s_t+1 for the next step.
            state = next_state
            if(len(agent.memory) > BATCH_SIZE):
                # Step 5: Train the Q deep network with replay experience.
                agent.replay(action_model)

    #Playing the game
    print("Playing the game")
    tot_reward = 0.0
    #action_model.load_weights("weights.h5")
    done = False
    state = env.reset() 
    state  = agent.add_to_stack(state, True)
    while not done:
        env.render()    
        # Normalizes the states and reshapes it so it has 4 dimensions (4 dimensions are required by the model)            
        state = state/255
        state = state.reshape((1, *INPUT_SHAPE)) 
        # Uses the model to predict the q values and chooses the one that has the maximum value. 
        choice = action_model.predict(state)                  
        index = np.argmax(choice)
        action = agent.possible_actions[index]  
        # Peforms the action chosen by model       
        next_state, reward, done, _ = env.step(action)
        # Add the new frame to the stack    
        next_state = agent.add_to_stack(next_state)
        tot_reward += reward
        state = next_state
    print('Game ended! Total reward: {}'.format(tot_reward))                 
