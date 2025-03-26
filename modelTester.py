import torch
from model import BOTModel as BOT
from generateSample import Trajectory
import pandas as pd



if __name__ == '__main__' :
    model = BOT()
    model.load_state_dict(torch.load("model2.pth"))
    model.eval()
    print(model)

    # Test the model
    # traj = Trajectory([12,-1,2,-1],0.01,[1,0,2,0])
    # traj.move([0]*15)
    # traj.write_to_csv("testModel.csv")

    df = pd.read_csv("testModel.csv")
    positions = torch.tensor(df[['pos_x', 'pos_y']].values, dtype=torch.float32)  # Position du capteur
    angles = torch.tensor(df['angle'].values, dtype=torch.float32).unsqueeze(-1)  # Angles observés
    X_eval = torch.cat([angles, positions], dim=1)
    score = model(X_eval)

    print(score)
    