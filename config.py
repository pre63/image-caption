import os
import torch

def configs(vocab):
        embed_sizes = [256, 512, 768, 1024, 2048]
        hidden_sizes = [256, 512, 768, 1024, 2048]
        num_layers = [1]
        learning_rates = [0.001, 0.0001, 0.00001, 0.005, 0.0005, 0.00005]
        num_epochs = [3, 10, 15, 60]
        configs = []
        for embed_size in embed_sizes:
                for hidden_size in hidden_sizes:
                        for num_layer in num_layers:
                                for learning_rate in learning_rates:
                                        for epoch in num_epochs:
                                          slug = f'{embed_size}_{hidden_size}_{num_layer}_{learning_rate}_{epoch}'
                                          configs.append((slug, epoch, embed_size, hidden_size, len(vocab), num_layer, learning_rate))
        # print the number of vatiations and all the configs
        print(f'Number of variations: {len(configs)}')

        return configs

# wrire a report fucntion that writes to the disk the report for the slug as a file name
def report(slug, line):
        with open(f'reports/{slug}.txt', 'a') as f:
                f.write(line)
                
                
