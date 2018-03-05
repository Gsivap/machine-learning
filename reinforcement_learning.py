import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from random import *
import random
from collections import deque

#Using a deque as replay memory
replay_memory = deque([], 10000 )

#Defining all the hyperparameters
iteration = 0
dis_rate = 0.95
save_modit = 20
train_start = 1500
done = True
dict_action = { 0: 4, 1:5 }
eps_max = 1
eps_min = 0.05
training_step = 0
batch_size = 32
max_training = 1000000
episode_reward = 0

def frame_preprocessing(frame):
    '''
    Function takes in a frame, resizes it, converts it into grayscale and normalizes the frame between -1 and 1
    '''
    frame = frame[22:210:2,::2] #resize the frame
    frame = frame.mean(axis = 2) #convert into grayscale
    back_color = frame[20,40] # find the background color
    frame[frame == back_color] = 0 #make background zero for better constrast
    x = (frame - 128)/128 # normalize the image between -1 and 1 to make it suitable for neural net
    
    return x
    
def to_variable(x,requires_grad = False,cuda = False):

    """
    Convert numpy array into a pytorch tensor
    
    """    
    x  =  torch.from_numpy(x)
    if(cuda):
        x = x.cuda()
    x = Variable(x, requires_grad = requires_grad)

    return x
    
    
def sample_replaymem(batch_size):

    """
    Samples are drawn from the replay memory.
    
    """
    
    
    training_list = sample(replay_memory,batch_size)
    x_train = np.zeros((batch_size,4,94,80))
    action = np.zeros((batch_size,1))
    reward = np.zeros((batch_size,1))
    next_state = np.zeros((batch_size,4,94,80))
    done = np.zeros((batch_size))
    
    for i in range(0,len(training_list)):
        x_train[i,:,:,:] = training_list[i][0]
        action[i] = training_list[i][1]
        reward[i] =  training_list[i][2]
        next_state[i,:,:,:] = training_list[i][3]
        done = training_list[i][4]  
    return x_train,action,reward,next_state,done
        
        
def choose_action(out_net,training_step):


    epsilon = max(eps_max,eps_max - (eps_max-eps_min)/50000 * training_step)
    if random.random() < epsilon:
        action = np.random.choice([4,5])
    else:
        action = np.argmax(out_net,axis =1)
        action = dict_action[action[0]]
        
    return action
    


class Neural_net(nn.Module):
    """
    This class defines of the neural network that we will use to predict the Q-values.
    It takes 4 frames stacked up as input state and predicts the Q-value for that particular state
    """
    
    def __init__(self):
        super(Neural_net, self).__init__()
        self.conv1 = nn.Conv2d(4,32,8,2)
        self.conv2 = nn.Conv2d(32,64,4,2)
        self.conv3 = nn.Conv2d(64,64,3,1)
        self.fc1 = nn.Linear(18240,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,2)
        
    def forward(self,x):
        
        x = (F.relu(self.conv1(x)))
        x = (F.relu(self.conv2(x)))
        x = (F.relu(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = (F.relu(self.fc1(x)))
        x = (F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Neural_net()
net = net.double()
cuda = False
if torch.cuda.is_available():
    
    cuda = True
    net = net.cuda()

env = gym.make('AirRaid-v0')








while (True):
    iteration += 1
    if training_step > max_training:
        break
        
    if done:
        obs =  env.reset()
        print("Episode reward{} iterations:{}".format(episode_reward,iteration))
        episode_reward = 0
        #skip the start of the game to get different starting state each time
        for skip in range(20):
            obs,reward,done,_ = env.step(np.random.choice([4,5]))
            if (skip == 19):
                    frame = frame_preprocessing(obs)
                    cur_state = np.stack((frame,frame,frame,frame),axis = 2)
                    cur_state = np.transpose(cur_state,(2,0,1))
    
    
    # Get the action prediction 
    inp = cur_state.reshape((1,4,94,80))
    inp = to_variable(inp,cuda = cuda)
    out_net = net(inp)
    action = choose_action(out_net,training_step)
     
    # Take the predicted action and find the next state   
    nxt_obs,cur_reward,done,_= env.step(action) 
    nxt_obs = frame_preprocessing(nxt_obs)
    nxt_obs = nxt_obs.reshape((1,94,80))


    #Store [current state,action,reward,next state] in replay memory
    nxt_state = np.append(nxt_obs,cur_state[:3,:,:],axis = 0)
    episode_reward += cur_reward
    replay_memory.append([cur_state,action,cur_reward,nxt_state,done])
    cur_state = nxt_state
    
    if iteration < train_start:
        continue
        

    #sample from replay memory
    training_step = training_step + 1
    x_train,action,reward,next_state,status = sample_replaymem(batch_size)
        
        
        
    #find the max_qvalue for the next state
    max_qvals_next = np.max(net(to_variable(next_state,cuda = cuda)).cpu().data.numpy(),axis = 1, keepdims = True)
    
        
    #calculate the target q_values
    target_q_values = reward + (1-status)*dis_rate*max_qvals_next
    
        
    #find the actual prediction of the network
    predicted_q_values = net(to_variable(x_train,cuda = cuda))
    action_toupdate = np.argmax(predicted_q_values.cpu().data.numpy(),axis = 1)
    action_toupdate = action_toupdate.tolist()
    
    target = net(to_variable(x_train,cuda = cuda)).cpu().data.numpy()
    target[action_toupdate,:] = target_q_values

        
    #criteria and optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    optimizer.zero_grad()
    
    loss = F.smooth_l1_loss(predicted_q_values,to_variable(target,cuda = cuda))
    

    #optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    loss.backward()     
    optimizer.step()
        
    if (iteration%1000 == 0):
        torch.save(net.state_dict(),'re_model/model.pt')
        
        
print("Training Complete")

"""
Now "model.pt" can be loaded and used to play the game.

"""

"""
TODO:
Upload results of playing the game

"""

        
        
        
        
