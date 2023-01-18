import torch.nn.functional as F

def train(iter, train_set, network, optimizer, size, log_interval: int = 10):
    network.train()
    
    for batch_idx, (data, target) in enumerate(train_set):
        optimizer.zero_grad()

        # data[iter * batch:(iter + 1) * batch]
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print('Train Iter: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  iter, batch_idx * len(data), len(train_set.dataset),
                  100. * batch_idx / len(train_set), loss.item()))