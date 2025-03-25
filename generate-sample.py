#Code that generate sample path
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random as r

#Every vector position is in the followong form [x,vx,y,vy]

class Trajectory:
    def __init__(self,starting_position,measure_noise,target_position):
        self.position = [starting_position]
        self.noise = measure_noise
        self.target_position = target_position
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

        return list(np.dot(T,self.position[-1]))
    
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
    
    def display(self) :
        fig,ax = plt.subplots()
        ax.scatter(self.target_position[0],self.target_position[2],c='r')
        ax.scatter(self.position[0][0],self.position[0][2],c='b')
        ax.plot([a[0] for a in self.position] ,[a[2] for a in self.position],c='b')
        plt.show()
        return

    def write_to_csv(self,filename) :
       # Create a DataFrame from the trajectory data
            data = {
                "pos_x": [pos[0] for pos in self.position],
                "pos_y": [pos[2] for pos in self.position],  
                "angle": self.measurements,
                "target_x": [self.target_position[0]] * len(self.position),
                "target_y": [self.target_position[2]] * len(self.position),
            }

            df = pd.DataFrame(data)

            # Check if file exists
            if not os.path.isfile(filename):
                df.to_csv(filename, mode="w", index=False)
            else:
                df.to_csv(filename, mode="a", index=False, header=False)


def generate_sample_data(nb_simu,length_simu,filename) :
    action = [0]*length_simu

    for i in range(nb_simu) :
        target_pos = [np.random.uniform(-5,5,size=1)[0],0,np.random.uniform(-5,5,size=1)[0],0]
        test = Trajectory([12,-1,2,-1],0.01,target_pos)
        test.move(action)
        test.write_to_csv(filename)


if __name__ == '__main__':
    filename = "training.csv"
    length_simu = 15
    nb_simu = 100

    action_space = np.linspace(-2*np.pi/5,2*np.pi/5,5)
    

