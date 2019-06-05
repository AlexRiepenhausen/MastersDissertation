import torch
import torch.nn as nn
from lstm.lstm import LSTMModel, LSTMCell
from tqdm import tqdm
from lstm.dataReaderVec import VectorDataset
from torch.autograd import Variable

class LSTMTrainer:

    def __init__(self, vec_files, label_file, learning_rate, input_dim, hidden_dim, layer_dim, output_dim):

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim  = layer_dim
        self.output_dim = output_dim

        self.model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)

        if torch.cuda.is_available():
            self.model.cuda()

        self.criterion  = nn.CrossEntropyLoss()

        self.learning_rate = learning_rate
        self.optimiser = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.train_loader = VectorDataset(vec_files, label_file)
        self.test_loader   = VectorDataset(vec_files, label_file) # identical to train_loader for now


    def evaluateModel(self,i,seq_dim):

        correct = 0
        total = 0
        for j, (vector_doc, label) in enumerate(self.test_loader):

            if torch.cuda.is_available():
                vector_doc = Variable(vector_doc.view(-1, seq_dim, self.input_dim).cuda())
            else:
                vector_doc = Variable(vector_doc.view(-1, seq_dim, self.input_dim))

            # Forward pass only to get logits/output
            outputs = self.model(vector_doc)

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs, 0)

            # Total number of labels
            total += 1

            if torch.cuda.is_available():
                _, label = torch.max(label,0)
                correct += (predicted.cpu() == label.cpu()).sum()
            else:
                correct += (predicted == label).sum()

            if j % 100 == 0 and j > 0:
                break

        return 100 * correct / total


    def train(self, num_epochs, seq_dim):

        accuracies = []

        for epoch in tqdm(range(num_epochs)):
            for i, (vector_doc, label) in enumerate(self.train_loader):

                loss_list = []

                if torch.cuda.is_available():
                    vector_doc = Variable(vector_doc.view(-1, seq_dim, self.input_dim).cuda())
                    label      = Variable(label.cuda())
                else:
                    vector_doc = Variable(vector_doc.view(-1, seq_dim, self.input_dim))
                    label      = Variable(label)

                # Clear gradients w.r.t. parameters
                self.optimiser.zero_grad()

                # Forward pass to get output/logits
                # outputs.size() --> 1, 3
                outputs = self.model(vector_doc)

                # Calculate Loss: softmax --> cross entropy loss
                loss = self.criterion(outputs.view(-1,1), label)

                if torch.cuda.is_available():
                    loss.cuda()
                # Getting gradients w.r.t. parameters
                loss.backward()

                # Updating parameters
                self.optimiser.step()

                loss_list.append(loss.item())

                if i % 1000 == 0:
                    accuracies.append(self.evaluateModel(i,seq_dim))
                    break

        return accuracies