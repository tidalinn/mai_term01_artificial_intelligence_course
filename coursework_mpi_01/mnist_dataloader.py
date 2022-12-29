import torchvision
import torch.utils.data as data

def mnist_dataloader(train: bool = True, 
                     batch_size: int = 64):
    
    """Dataloader of the MNIST dataset
       
       Args:
           train (bool, optional): train of test. Defaults to True
           batch_size (int, optional): batch size. Defaults to 64
       
       Return:
           Loaded MNIST dataset
    """
    
    return data.DataLoader(
                torchvision.datasets.MNIST("./", 
                    train=train, 
                    download=True, 
                    transform=torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
                    shuffle=True,
                    num_workers=1,
                    batch_size=batch_size)