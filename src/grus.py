'''
This module contains an implementation of a GRU cell in pytorch, as well as
some utility functions. See the GRUs_DEMO Notebook for a demonstration. The intent of
this implementation is to solidify my understanding of the inner working of GRUs. For
more serious applications, use the pytorch native implemnetation.
'''
import torch
from torch import nn, tanh, sigmoid


NON_LINEARITIES = {
    'relu' : nn.ReLU(),
    'tanh' : nn.Tanh(),
    'softmax': nn.Softmax(dim=1)
}


class GRU_Cell(nn.Module):
    """
    A pytorch implementation of a GRU Cell.
    """
    def __init__(self, hidden_size, input_size, output_size, output_activation='softmax'):
        '''
        hidden_size: dim of internal representation of state
        input_size: dim of most basic vector (e.g. vocab size)
        output_size: dim of output vector
        batch_size: number of examples in a batch
        '''
        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        # update gate parameters
        self.W_u = nn.Linear(input_size, hidden_size,  bias=True)  # input transformation matrix
        self.K_u = nn.Linear(hidden_size, hidden_size, bias=True) # state transformation matrix

        # reset gate parameters
        self.W_r = nn.Linear(input_size, hidden_size, bias=True) # input transformation matrix
        self.K_r = nn.Linear(hidden_size, hidden_size, bias=True) # state transformation matrix

        # new state parameters
        self.W_h = nn.Linear(input_size, hidden_size, bias=True) # input transformation matrix
        self.K_h = nn.Linear(hidden_size, hidden_size, bias=True) # input transformation matrix

        # state to output mapping parameters
        self.W_o = nn.Linear(hidden_size, output_size, bias=True)
        self.output_activation = NON_LINEARITIES[output_activation]


    def forward(self, x_t, h_prev):
        '''
        x_t: (input_size, batch_size)
        '''
        update_gate = sigmoid(self.W_u(x_t) + self.K_u(h_prev))
        reset_gate = sigmoid(self.W_r(x_t) + self.K_r(h_prev))
        h_candidate = tanh(self.W_h(x_t) + reset_gate * self.K_h(h_prev))
        h_new = update_gate * h_prev + (1-update_gate) * h_candidate
        output = self.output_activation(self.W_o(h_new))
        h_new = h_new.detach()
        return output, h_new


def cross_entropy(y_pred, y_test):
    '''
    '''
    eps=1e-7
    return -torch.mean(torch.sum(y_test * torch.log(y_pred+eps), dim=0), dim=0)


def compute_loss(output, target, loss_function):
    '''
    '''
    total_loss = 0
    for y_pred, y_test in zip(output, target):
        batch_loss = loss_function(y_pred, y_test)
        total_loss += batch_loss
    return total_loss


def train_gru(model, loader, epochs, loss_function, optimizer, device, verbose=True):
    '''

    '''

    for epoch in range(epochs):
        current_state = torch.zeros(loader.batch_size, model.hidden_size)
        current_state.to(device)

        for x_train, y_train in loader:
            if x_train.shape[0] != loader.batch_size:
                continue

            optimizer.zero_grad()

            activation, current_state = model(x_train, current_state)

            loss = compute_loss(activation, y_train, loss_function)

            loss.backward(retain_graph=True)

            optimizer.step()

        if epoch % 10 == 0:
            print(f'epoch {epoch} loss = {loss:.2f}')

    return current_state
