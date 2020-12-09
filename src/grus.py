import torch
from torch import nn
from torch import sigmoid, tanh, mm, randn, relu
from torch.autograd import Variable


class GRU_Cell:
    """
    A pytorch implementation of a GRU Cell.
    """
    def __init__(self, hidden_size, input_size, batch_size, output_size):
        '''
        hidden_size: dim of internal representation of state
        input_size: dim of most basic vector (e.g. vocab size)
        output_size: dim of output vector
        batch_size: number of examples in a batch
        '''

        self.hidden_size = hidden_size
        self.input_size = input_size

        # update gate parameters
        self.W_u = Variable(randn(hidden_size, input_size), requires_grad=True) # input transformation matrix
        self.K_u = Variable(randn(hidden_size, hidden_size), requires_grad=True) # state transformation matrix

        # reset gate parameters
        self.W_r = Variable(randn(hidden_size, input_size), requires_grad=True) # input transformation matrix
        self.K_r = Variable(randn(hidden_size, hidden_size), requires_grad=True) # state transformation matrix

        # new state parameters
        self.W_h = Variable(randn(hidden_size, input_size), requires_grad=True) # input transformation matrix
        self.K_h = Variable(randn(hidden_size, hidden_size), requires_grad=True) # input transformation matrix

        # state to output mapping parameters
        self.W_o = Variable(randn(output_size, hidden_size), requires_grad=True)


    def forward(self, x_t, h_prev):
        '''
        x_t: (input_size, batch_size)
        '''
        update_gate = sigmoid(mm(self.W_u, x_t) + mm(self.K_u, h_prev))
        reset_gate = sigmoid(mm(self.W_r, x_t) + mm(self.K_r, h_prev))
        h_candidate = tanh(mm(self.W_h, x_t ) + reset_gate * mm(self.K_h, h_prev))
        h_new = update_gate * h_prev + (1-update_gate) * h_candidate
        output = relu(mm(self.W_o, h_new))
        return output, h_new
