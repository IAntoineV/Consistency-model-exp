import torch
from torch.utils.data import Dataset


class LineDataset(Dataset):
    def __init__(self, len, x:torch.Tensor,y:torch.Tensor):
        super(LineDataset, self).__init__()
        self.len = len
        self.x = x
        self.y = y
        assert x.shape == y.shape
        shape  = list(x.shape)
        len_shape = len(shape)
        data = x.unsqueeze(0)+ (y-x).unsqueeze(0)*torch.rand([self.len]).view(-1, *[1]*len_shape )
        self.data = data

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        return self.data[idx]


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x = torch.Tensor([1,1])
    y = torch.Tensor([2,2])
    dataset = LineDataset(100,x,y)
    print(len(dataset))
    plt.scatter(dataset.data[:,0], dataset.data[:,1])
    plt.show()
