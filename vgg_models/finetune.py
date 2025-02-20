#name: finetune.py
import torch
import numpy as np
from ComprehensiveVGGPruner import prune_conv_layer
from operator import itemgetter
from heapq import nsmallest

class FilterPruner:
    def __init__(self, model):
        self.model = model
        self.reset()
        self.device = 'mps' if torch.backends.mps.is_available() \
            else 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def reset(self):
        self.filter_ranks = {}

    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

        activation_index = 0
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1

        return self.model.classifier(x.view(x.size(0), -1))

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]

        taylor = activation * grad
        # Get the average value for every filter, 
        # across all the other dimensions
        taylor = taylor.mean(dim=(0, 2, 3)).data


        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = \
                torch.FloatTensor(activation.size(1)).zero_()

        self.filter_ranks[activation_index] += taylor
        self.grad_index += 1

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v.cpu()

    def get_pruning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

        # After each of the k filters are pruned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (layer, filter, _) in filters_to_prune:
            if layer not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[layer] = []
            filters_to_prune_per_layer[layer].append(filter)

        for layer in filters_to_prune_per_layer:
            filters_to_prune_per_layer[layer] = sorted(filters_to_prune_per_layer[layer])
            for i in range(len(filters_to_prune_per_layer[layer])):
                filters_to_prune_per_layer[layer][i] = filters_to_prune_per_layer[layer][i] - i

        filters_to_prune = []
        for layer in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[layer]:
                filters_to_prune.append((layer, i))

        return filters_to_prune

class PruningFineTuner:
    def __init__(self, train_path, test_path, model):
        self.train_data_loader = None # dataset.train_loader(train_path)
        self.test_data_loader = None # dataset.test_loader(test_path)

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.pruner = FilterPruner(self.model) 
        self.model.train()

    def train(self, optimizer = None, epochs=10):
        if optimizer is None:
            optimizer = torch.optim.SGD(self.model.classifier.parameters(), lr=0.0001, momentum=0.9)

        for i in range(epochs):
            print("Epoch: ", i)
            self.train_epoch(optimizer)
            self.test(self.model)
        print("Finished fine tuning.")
        
    def test(self, model):
        
        self.model.eval()
        correct = 0
        total = 0

        for batch, label in self.test_data_loader:
            batch, label = batch.to(self.device), label.to(self.device)
            output = model(batch)
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label).sum()
            total += label.size(0)

        print("Accuracy :", float(correct) / total)

        self.model.train()

    def train_batch(self, optimizer, batch, label, rank_filters):

        self.model.zero_grad()
        input = torch.Variable(batch)

        if rank_filters:
            output = self.pruner.forward(input)
            self.criterion(output, torch.Variable(label)).backward()
        else:
            self.criterion(self.model(input), label).backward()
            optimizer.step()

    def train_epoch(self, optimizer = None, rank_filters = False):
        for batch, label in self.train_data_loader:
            self.train_batch(optimizer, batch, label, rank_filters)

    def get_candidates_to_prune(self, num_filters_to_prune):
        self.pruner.reset()
        self.train_epoch(rank_filters = True)
        self.pruner.normalize_ranks_per_layer()
        return self.pruner.get_pruning_plan(num_filters_to_prune)
        
    def total_num_filters(self):
        filters = 0
        for _, module in self.model.features._modules.items():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                filters = filters + module.out_channels
        return filters

    def prune(self, prune_amount):
        #Get the accuracy before pruning
        self.test(self.model)
        self.model.train()

        #Make sure all the layers are trainable
        for param in self.model.features.parameters():
            param.requires_grad = True

        number_of_filters = self.total_num_filters()
        num_filters_to_prune_per_iteration = 512
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)

        iterations = int(iterations * (prune_amount))

        print(f"Number of pruning iterations to reduce {prune_amount*100}\% filters:", iterations)

        for _ in range(iterations):
            print("Ranking filters.. ")
            prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
            layers_pruned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_pruned:
                    layers_pruned[layer_index] = 0
                layers_pruned[layer_index] = layers_pruned[layer_index] + 1 

            print("Layers that will be pruned", layers_pruned)
            print("Pruning filters.. ")
            model = self.model.cpu()
            for layer_index, filter_index in prune_targets:
                model = prune_conv_layer(model, layer_index, filter_index)

            self.model = model.to(self.device)

            message = f"{str(100 * float(self.total_num_filters()) / number_of_filters)}%"
            print("Filters pruned", message)
            self.test(self.model)
            print("Fine tuning to recover from pruning iteration.")
            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            self.train(optimizer, epochs = 10)


        print("Finished. Going to fine tune the model a bit more")
        self.train(optimizer, epochs=10)
        self.test(self.model)
        