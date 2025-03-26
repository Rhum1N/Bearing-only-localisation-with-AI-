import pandas as pd
import torch
import torch.nn as nn

class BOTModel(nn.Module) :
    def __init__(self,input_size=3,hidden_size=64,output_size=2) :
        super(BOTModel,self).__init__()
        self.rnn = nn.LSTM(input_size,hidden_size,batch_first=True)
        self.fc = nn.Linear(hidden_size,output_size)

    def forward(self, x) :
        _, (hn, _) = self.rnn(x)
        x = self.fc(hn.squeeze(0))
        return x


def create_sequences(positions,angles,target_positions,seq_length = 15) :
    X,Y = [],[]

    for i in range(len(angles)-seq_length) :
        angle_seq = angles[i:i+seq_length].unsqueeze(-1)
        sensor_seq = positions[i:i+seq_length]  
        X.append(torch.cat([angle_seq, sensor_seq],dim=1))  # Concaténer (θ, x_s, y_s)
        Y.append(target_positions[i+seq_length])  # Position cible
    return torch.stack(X), torch.stack(Y)

if __name__ == '__main__' :

    #get the data
    df = pd.read_csv("training.csv")

    positions = torch.tensor(df[['pos_x', 'pos_y']].values, dtype=torch.float32)  # Position du capteur
    angles = torch.tensor(df['angle'].values, dtype=torch.float32)  # Angles observés
    target_positions = torch.tensor(df[['target_x', 'target_y']].values, dtype=torch.float32)  # Position cible

    X_train, Y_train = create_sequences(positions, angles, target_positions, seq_length=15)
    model = BOTModel(input_size=3, hidden_size=64, output_size=2)
    
    
    # Train the model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(2000) :

        optimizer.zero_grad()
        output = model(X_train)
        Loss = criterion(output,Y_train)
        Loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss {Loss.item()}")
    
    model.eval()
    print(model)


    #save model 
    torch.save(model.state_dict(),"model1.pth")
