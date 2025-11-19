import torch;
import torch.nn as nn;

class LSTM_Classifier(nn.Module):
    def __init__(self, input_size = 28, hidden_size = 128, num_layers = 2, num_classes = 10):
        super(LSTM_Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, data):
        #data: (batch_size, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, data.size(0), self.hidden_size).to(data.device)
        c0 = torch.zeros(self.num_layers, data.size(0), self.hidden_size).to(data.device)

        out, _ = self.lstm(data, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out