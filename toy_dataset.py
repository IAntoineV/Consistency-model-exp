import torch
from torch.utils.data import Dataset


class LineDataset(Dataset):
    def __init__(self, length, x:torch.Tensor,y:torch.Tensor):
        super(LineDataset, self).__init__()
        self.length = length
        self.x = x
        self.y = y
        assert x.shape == y.shape
        shape  = list(x.shape)
        len_shape = len(shape)
        data = x.unsqueeze(0)+ (y-x).unsqueeze(0)*torch.rand([self.length]).view(-1, *[1]*len_shape )
        self.data = data
        self.shape= shape

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        return self.data[idx]


class Circle2DDataset(Dataset):
    def __init__(self, length, center:torch.Tensor,r):
        super(Circle2DDataset, self).__init__()
        self.length = length
        self.center = center
        self.r = r
        thetas = 2*torch.pi * torch.rand([self.length])
        x,y = torch.cos(thetas), torch.sin(thetas)
        data = torch.stack([x,y], dim=-1)
        self.data = data
        self.shape = list(x.shape)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        return self.data[idx]



# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x = torch.Tensor([1,1])
    y = torch.Tensor([2,2])
    #dataset = LineDataset(100,x,y)
    dataset = Circle2DDataset(100, x, 2)
    print(len(dataset))
    plt.scatter(dataset.data[:,0], dataset.data[:,1])
    plt.show()
