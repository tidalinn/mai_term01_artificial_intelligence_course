from mpi4py import MPI

import torch
import torch.optim as optim

from mnist_dataloader import mnist_dataloader
from slice_dataset import slice_dataset
from mnist_network import MNISTNetwork
from train import train
from test import test

if __name__ == '__main__':

    train_mnist = mnist_dataloader()
    test_mnist = mnist_dataloader(False)

    results_test = []

    learn_r = 0.01
    moment = 0.5

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    network = MNISTNetwork()
    optimizer = optim.SGD(network.parameters(), lr=learn_r, momentum=moment)


    for i in range(size):
        print(f'\nRank {rank}')
        print(f'[ = = = = = Iter {i} = = = = = ]')

        if rank == 0:
            [comm.send(network.state_dict(), k) for k in range(1, size)]
            train(i, train_mnist, network, optimizer, size)
            results_test.append(test(test_mnist, network))

        else:
            network.load_state_dict(comm.recv())
            train(i, train_mnist, network, optimizer, size)
            results_test.append(test(test_mnist, network))
        

    print(results_test)

    with open('./results_test.txt', 'w') as file:
        result_to_save = ', '.join(str(value[0]) 
                                for value in [result['accuracy'] 
                                for result in results_test])
        file.write(result_to_save)

MPI.Finalize