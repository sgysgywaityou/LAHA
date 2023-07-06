import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from .tidy_data import form_data
class MyDataset(Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
    def __len__(self):
        return self.data_tensor.size(0)
    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

def load_data(file_path,b):
    sentence_data,label_data ,num= form_data(file_path)
    num_list_sentence=[]
    num_list_label=[]
    for i in range(num):
        num_list_sentence.append(2*i)
        num_list_label.append(i)
    num_tensor_sentence=torch.tensor(num_list_sentence)
    num_tensor_label=torch.tensor(num_list_label)
    my_dataset = MyDataset(num_tensor_sentence, num_tensor_label)
    tensor_dataloader = DataLoader(dataset=my_dataset,
                                   batch_size=b,
                                   shuffle=True,
                                   num_workers=0,
                                   drop_last=True)
    return tensor_dataloader,sentence_data,label_data,num