import torch
import torch.nn as nn
from lstm.lstm import LSTMModel
from tqdm import tqdm
from lstm.dataReaderVec import VectorDataset
from torch.autograd import Variable
from utilities.utilities import weightInit

class LSTMTrainer:

    def __init__(self, train_files, train_labels, test_files, test_labels, learning_rate, iterations_per_epoch,
                 input_dim, category, hidden_dim, layer_dim, output_dim):

        self.input_dim  = input_dim
        self.category   = category
        self.hidden_dim = hidden_dim
        self.layer_dim  = layer_dim
        self.output_dim = output_dim

        self.iterations_per_epoch = iterations_per_epoch

        self.model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)

        self.criterion  = nn.CrossEntropyLoss()

        self.learning_rate = learning_rate
        self.optimiser     = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.train_loader  = VectorDataset(train_files, train_labels, category)
        self.test_loader   = VectorDataset(test_files, test_labels, category)

        self.to_string = "lr_{}_ipe_{}_in_{}_ct_{}_hd_{}_ly_{}_out_{}".format(learning_rate,
                                                                       iterations_per_epoch,
                                                                       input_dim,
                                                                       category,
                                                                       hidden_dim,
                                                                       layer_dim,
                                                                       output_dim)                                                                       

    def initDevice(self):
        if torch.cuda.is_available():
            self.model.cuda()


    def initWeights(self, init, saved_model_path=None):
        if init == weightInit.load:
            self.model.load_state_dict(torch.load(saved_model_path))
            self.model.eval()
        elif init == weightInit.fromScratch:
            pass  # -> already initialised
        else:
            print("Specified weight initialisation not supported. Stop.")
            exit(0)


    def evaluateModel(self, test_samples=100, test=True):

        correct = 0
        total = 0
        
        labels_true = list()
        labels_pred = list()
        
        loader = self.test_loader
        if test == False:
            loader = self.train_loader
        
        for j, (vector_doc, label) in enumerate(loader):
        
            if torch.cuda.is_available():
                vector_doc = Variable(vector_doc.view(-1, len(vector_doc), self.input_dim).cuda())
                label = Variable(label.cuda())
            else:
                vector_doc = Variable(vector_doc.view(-1, len(vector_doc), self.input_dim))
                label = Variable(label)

            # Forward pass only to get logits/output
            outputs = self.model(vector_doc)

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs, 0)

            # Total number of labels
            total += 1

            if torch.cuda.is_available():
                if predicted.cpu() == label.squeeze(dim=0).cpu():
                    correct += 1
            else:
                if predicted.cpu() == label.squeeze(dim=0).cpu():
                    correct += 1

            labels_true.append(label.squeeze(dim=0).cpu())
            labels_pred.append(predicted.cpu())

            if j == test_samples-1:
                accuracy = float(correct) / float(total)
                return (labels_true, labels_pred), accuracy


    def train(self, num_epochs, compute_accuracies, test_samples=100, init=weightInit.fromScratch, model_path=None):

        losses     = []
        accuracies = []
        parcel     = []

        self.initWeights(init, saved_model_path=model_path)
        self.initDevice()

        for epoch in tqdm(range(num_epochs)):

            avg_loss = 0.0

            for i, (vector_doc, label) in enumerate(self.train_loader):
                
                if torch.cuda.is_available():
                    vector_doc = Variable(vector_doc.view(-1, len(vector_doc), self.input_dim).cuda())
                    label = Variable(label.cuda())
                else:
                    vector_doc = Variable(vector_doc.view(-1, len(vector_doc), self.input_dim))
                    label = Variable(label)
                
                # Clear gradients w.r.t. parameters
                self.optimiser.zero_grad()
                
                # Forward pass to get output/logits
                # outputs.size() --> 1, 3
                
                outputs = self.model(vector_doc)
                
                # Calculate Loss: softmax --> cross entropy loss
                loss = self.criterion(outputs.unsqueeze(dim=0), label.unsqueeze(dim=0))
                
                if torch.cuda.is_available():
                    loss.cuda()
                # Getting gradients w.r.t. parameters
                loss.backward()
      
                # Updating parameters
                self.optimiser.step()

                avg_loss += loss.item()

                # save losses and accuracies every self.iterations_per_epoch
                if self.runEvaluation(i):
                    losses.append(avg_loss/self.iterations_per_epoch)
                    if compute_accuracies==True:
                        _, accuracy = self.evaluateModel(test_samples, test=True)
                        accuracies.append(accuracy)
                    break

        parcel.append(losses)
        if compute_accuracies == True:
            parcel.append(accuracies)

        return parcel


    # check if it is necessary to run evaluation of accuracy
    def runEvaluation(self,iter):

        if self.iterations_per_epoch == 1:
            return True
        else:
            if iter % (self.iterations_per_epoch - 1) == 0 and iter > 0:
                return True
            return False