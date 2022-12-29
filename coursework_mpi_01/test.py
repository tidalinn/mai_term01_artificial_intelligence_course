import torch
import torch.nn.functional as F

def test(test_set, network):
    network.eval()
    test_loss = 0
    correct = 0
    results = {'avg_loss': [],
               'accuracy': []}
    
    with torch.no_grad():
        for data, target in test_set:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            
    test_loss /= len(test_set.dataset)
    
    results['avg_loss'].append(test_loss)
    results['accuracy'].append(100. * correct / len(test_set.dataset))

    return results