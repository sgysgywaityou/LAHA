import torch
import torch.nn.functional as F
import random
import warnings
from torch import nn

torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings("ignore")

class MultiHeadAttn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.multi_attn = nn.MultiheadAttention(embed_dim=768, num_heads=12, batch_first=True)

    def forward(self, q, k,v):
        mv_1 = self.multi_attn(q,k, v)
        return mv_1

class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path=None):
        if path is None:
            raise ValueError('Please specify the saving road!!!')
        torch.save(self.state_dict(), path)
        return path

class AttentionAndCombineNet(BasicModule):
    def __init__(self,m,p,x,y,r,n,batch_size,num_of_attention_head,num_train,):
        super(AttentionAndCombineNet, self).__init__()
        self.m = m
        self.x = x
        self.n=n
        self.r = r
        self.p=p
        self.batch_size = batch_size

        self.multiAttn1 = MultiHeadAttn()
        self.multiAttn2 = MultiHeadAttn()
        self.multiAttn3 = MultiHeadAttn()

        self.weight1=random.random()
        self.W1 = torch.nn.Linear(n, x)
        self.W2 = torch.nn.Linear(x, p)
        self.W3 = torch.nn.Linear(n, x)
        self.W4 = torch.nn.Linear(x, p)
        self.Wful = torch.nn.Linear(n * p, y)
        self.output_layer = torch.nn.Linear(y, m)

        self.batch_idx_max = num_train // batch_size
        self.batch_idx = 0
        self.CO_collection = []
        self.CO_collection_tmp = []

    def get_CosSim_OneByOne(self,input1, input2):
        result_list = []
        final_list = []
        for b in range(self.batch_size):
            target_n = input1[b]
            behaviored = input2[b]
            for i in range(target_n.size(0)):
                attention_distribution = []
                for j in range(behaviored.size(0)):
                    attention_score = torch.cosine_similarity(
                        target_n[i], behaviored[j].view(
                            1, -1), dim=1)
                    attention_distribution.append(attention_score)
                attention_distribution = torch.Tensor(attention_distribution)
                result_list.append(attention_distribution)
            result = torch.stack(result_list, dim=0)
            result_list.clear()
            final_list.append(result)
        final_result = torch.stack(final_list, dim=0)
        return final_result

    def get_embeddings_after_NVD(self,VC,VD,alpha):
        for i in range(VC.size(0)):
            for j in range(VC.size(1)):
                vc_j=0
                for k in range(VC.size(2)):
                    vc_j+=VC[i][j][k]
                vc_j=vc_j/VC.size(2)
                if vc_j<alpha:
                    for w in range(VD.size(2)):
                        VD[i][j][w]=0
        return VD

    def forward(self, NVD,LA,mode,now_epoch,batch_idx):
        if(mode==0):
            DSA=self.multiAttn1(NVD,NVD,NVD)[0]
            co_att_lab_doc=self.multiAttn2(DSA,LA,LA)[0]
            co_att_doc_lab=self.multiAttn3(LA,DSA,DSA)[0]
            weight1 = self.weight1
            weight2 = 1 - weight1
            co_t = weight1 *co_att_lab_doc + weight2 * co_att_doc_lab
            if self.r == 1:
                if now_epoch == 1:
                    print("this epoch is the first epoch")
                    HT_t = co_t
                    self.CO_collection_tmp.append(co_t)
                    print("self.CO_collection_tmp:",self.CO_collection_tmp)
                else:
                    if(batch_idx==0):
                        self.CO_collection=self.CO_collection_tmp
                        self.CO_collection_tmp=[]
                    print("this epoch is not the first epoch")
                    self.CO_collection_tmp.append(co_t)
                    print("self.CO_collection:",self.CO_collection)
                    co_tminus1 = (self.CO_collection[batch_idx]+co_t)/2
                    HT_t = F.softmax(self.W2(F.tanh(torch.bmm(co_t, F.tanh(self.W1(co_tminus1.transpose(1, 2)))))))
                    print("only fusing the last epoch")
            else:
                print("fusing all the history epochs")
                if now_epoch== 1:
                    print("this epoch is the first epoch")
                    HT_t = co_t
                    self.CO_collection.append(co_t)
                else:
                    print("this epoch is not the first epoch")
                    self.CO_collection[batch_idx]=self.CO_collection[batch_idx]+co_t
                    co_tminus1_ave=self.CO_collection[batch_idx]/now_epoch
                    HT_t = F.softmax(self.W2(F.tanh(torch.bmm(co_t, F.tanh(self.W1(co_tminus1_ave.transpose(1, 2)))))))
            HT_t_f = torch.flatten(HT_t, start_dim=1, end_dim=2)
            print("getting outputs...")
            outputs = torch.sigmoid(self.output_layer(F.relu(self.Wful(HT_t_f),inplace=False)))
            return outputs
        elif(mode==1):
            DSA=self.multiAttn1(NVD,NVD,NVD)[0]
            co_att_lab_doc=self.multiAttn2(DSA,LA,LA)[0]
            co_att_doc_lab=self.multiAttn3(LA,DSA,DSA)[0]
            weight1 = self.weight1
            weight2 = 1 - weight1
            co_t = weight1 * co_att_lab_doc + weight2 * co_att_doc_lab
            HT_t = F.softmax(self.W2(F.tanh(torch.bmm(co_t, F.tanh(self.W1(co_t.transpose(1, 2)))))))
            HT_t_f = torch.flatten(HT_t, start_dim=1, end_dim=2)
            outputs = torch.sigmoid(self.output_layer(F.relu(self.Wful(HT_t_f), inplace=False)))
            return outputs