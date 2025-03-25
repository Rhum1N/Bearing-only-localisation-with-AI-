#Code that generate sample path
import numpy as np
import pandas as pd

#Every vector position is in the followong form [x,vx,y,vy]

class Trajectory:
    def __init__(self,starting_position,measure_noise):
        self.position = [starting_position]
        self.noise = measure_noise
        self.target_position = [0,0,0,0]
        noise_angle = np.random.normal(0,measure_noise,size=1)[0]
        self.measurements = [np.arctan2(starting_position[2]-self.target_position[2],starting_position[0]-self.target_position[0])+noise_angle]
    
    #Compute the next position based on action 
    def __step(self,action) :
        timestep = 1
        if action != 0.0 :
            T = [
                [1, np.sin(action * timestep) / action, 0, -(1 - np.cos(action * timestep)) / action],
			    [0, np.cos(action * timestep), 0, -np.sin(action * timestep)],
			    [0, (1 - np.cos(action * timestep)) / action, 1, np.sin(action * timestep) / action],
			    [0, np.sin(action * timestep), 0, np.cos(action * timestep)
            ]]  
        else :
            T = [[1,timestep,0,0],[0,1,0,0],[0,0,1,timestep],[0,0,0,1]]

        return np.dot(T,self.position[-1])
    
    #Compute the noisy angle between the target and position
    def __measure_angle(self) :
        eps = np.random.normal(0,self.noise,size=1)[0]
        self.measurements.append(np.arctan2(self.position[-1][2]-self.target_position[2],self.position[-1][0]-self.target_position[0])+eps)


    #Simulate a trajectory taking actions in action_vector
    def move(self, action_vector) :
        for action in action_vector :
            self.position.append(self.__step(action))
            self.__measure_angle()
        return








if __name__ == '__main__':

    action_space = np.linspace(-2*np.pi/5,2*np.pi/5,5)

    test = Trajectory([12,-1,2,-1],0.01)
    test.move([2,2,2,2])
    print(test.measurements)
    print('hello world')

