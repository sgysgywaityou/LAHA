from model import AttentionAndCombineNet
from train import train
import torch
import load_data_index as index
import warnings


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')

n=500
m=54
p=768
b=64

train_loader,sentence_data_train,label_data_train,num_train= index.load_data('data/...', b)
test_loader,sentence_data_test,label_data_test,num_test= index.load_data('data/...', b)
print("load done")

def multilabel_classification(attention_model, train_loader, test_loader, sentence_data_train,label_data_train,sentence_data_test,label_data_test,num_train,epochs, GPU=False):
    loss = torch.nn.BCELoss()
    opt = torch.optim.Adam(attention_model.parameters(), lr=0.001, betas=(0.9, 0.99))
    train(attention_model, train_loader, test_loader, loss, opt, sentence_data_train,label_data_train,sentence_data_test,label_data_test,epochs,GPU)

attention_model = AttentionAndCombineNet(m=m,n=n,p=p,x=100,y=100,r=2,batch_size=b,num_of_attention_head=12,num_train=num_train)


multilabel_classification(attention_model, train_loader, test_loader, sentence_data_train,label_data_train,sentence_data_test,label_data_test,num_train, epochs=200)

attention_model.save('./MyModel.pth')