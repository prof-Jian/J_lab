import torch
a = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(a)
