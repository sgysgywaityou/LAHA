import numpy as np
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer, BertModel
import torch
import torch.nn.functional as F

list_NVD_train=[]
list_NVD_test=[]
list_LA_train=[]
list_LA_test=[]
y_one_hot_batch_list_all_train=[]
y_one_hot_batch_list_all_test=[]
def get_embedding_vector(text,max_len):
    tokenizer = BertTokenizer.from_pretrained(
        r".\data\bert_base_uncased_pytorch")
    model = BertModel.from_pretrained(
        r".\data\bert_base_uncased_pytorch")
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    with torch.no_grad():
        last_hidden_states = model(tokens_tensor)[0]
    token_embeddings = []
    if max_len==None:
        max_len = len(tokenized_text)
    if len(tokenized_text)>=max_len:
        for token_i in range(max_len):
            hidden_layers = []
            for layer_i in range(len(last_hidden_states)):
                vec = last_hidden_states[layer_i][0][token_i]
                hidden_layers.append(vec)
            token_embeddings.append(hidden_layers)
        summed_last_4_layers = [torch.sum(torch.stack(
            layer)[-4:], 0) for layer in token_embeddings]
        output = torch.stack(summed_last_4_layers, dim=0)
        return output
    else:
        for token_i in range(len(tokenized_text)):
            hidden_layers = []
            for layer_i in range(len(last_hidden_states)):
                vec = last_hidden_states[layer_i][0][token_i]
                hidden_layers.append(vec)
            token_embeddings.append(hidden_layers)
        summed_last_4_layers = [torch.sum(torch.stack(
            layer)[-4:], 0) for layer in token_embeddings]
        for plus_i in range(max_len-len(tokenized_text)):
            summed_last_4_layers.append(torch.zeros(768))
        output = torch.stack(summed_last_4_layers, dim=0)
        return output

def get_embedding_vector_label(one_hot_code):
    each_label_vector = torch.from_numpy(np.load('.\data\...'))
    tmp_list = []
    for i in range(len(one_hot_code)):
        if one_hot_code[i] == '1':
            # print(i)
            tmp_list.append(each_label_vector[i])
        else:
            tmp_list.append(torch.zeros(768))
    return torch.stack(tmp_list, 0)

def get_CosSim_OneByOne(input1, input2,batch_size):
    result_list = []
    final_list = []
    for b in range(batch_size):
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

def get_embeddings_after_NVD(VC,VD,alpha):
    for i in range(VC.size(0)):
        for j in range(VC.size(1)):
            vc_j=0
            for k in range(VC.size(2)):
                vc_j+=VC[i][j][k]
            vc_j=vc_j/VC.size(2)

            if vc_j<alpha:
                # print("delete")
                for w in range(VD.size(2)):
                    VD[i][j][w]=0
    return VD

def train(attention_model,train_loader,test_loader,criterion,opt,sentence_data_train,label_data_train,sentence_data_test,label_data_test,epochs = 5,GPU=False):
    if GPU:
        attention_model.cuda()

    for p in range(1):
        epoch_now=p+1
        print("Running EPOCH",1)
        train_loss = []
        prec_k = []
        ndcg_k = []

        for batch_idx, train in enumerate(tqdm(train_loader)):
            print("Epoch:",epoch_now)
            print("Batch Indec:",batch_idx)
            opt.zero_grad()
            x_index_list, y_index_list = train[0], train[1]

            x_embedding_list = []
            y_embedding_list = []
            y_one_hot_list_batch = []
            for i in range(len(x_index_list)):

                index=x_index_list[i]
                index=index/2
                index=int(index)

                x_words=sentence_data_train[index]
                y_words=label_data_train[y_index_list[i]].data.cpu()

                list_y_one_hot = []
                for i in range(len(y_words)):
                    if y_words[i] == '1':
                        list_y_one_hot.append(1)
                    else:
                        list_y_one_hot.append(0)

                y_one_hot_list_batch.append(list_y_one_hot)
                x_embedding_list.append(get_embedding_vector(x_words,500))
                y_embedding_list.append(get_embedding_vector_label(y_words))
            x = torch.stack(x_embedding_list, 0)
            y = torch.stack(y_embedding_list, 0)
            VD = x
            VL = y
            VC = F.softmax(get_CosSim_OneByOne(VD, VL,batch_size=2),
                               dim=1)
            NVD = get_embeddings_after_NVD(VC, VD, alpha=0.1)
            NVC = F.softmax(get_CosSim_OneByOne(NVD, VL,batch_size=2),
                              dim=1)
            LA = torch.bmm(NVC, VL)
            list_NVD_train.append(NVD)
            list_LA_train.append(LA)
            y_one_hot_batch = torch.from_numpy(np.array(y_one_hot_list_batch))
            y_one_hot_batch_list_all_train.append(y_one_hot_batch)
            y_pred= attention_model(NVD,LA,0,epoch_now,batch_idx)
            loss = criterion(y_pred, y_one_hot_batch.float()) / train_loader.batch_size
            loss.requires_grad_(True)
            loss.backward(retain_graph=True)
            opt.step()
            labels_cpu = y_one_hot_batch.data.cpu().float()
            pred_cpu = y_pred.data.cpu()
            prec = precision_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            prec_k.append(prec)
            ndcg = Ndcg_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            ndcg_k.append(ndcg)
            train_loss.append(float(loss))
        avg_loss = np.mean(train_loss)
        epoch_prec = np.array(prec_k).mean(axis=0)
        epoch_ndcg = np.array(ndcg_k).mean(axis=0)
        print("epoch %2d train end : avg_loss = %.4f" % (epoch_now, avg_loss))
        print("precision@1 : %.4f , precision@3 : %.4f , precision@5 : %.4f " % (epoch_prec[0], epoch_prec[2], epoch_prec[4]))
        print("ndcg@1 : %.4f , ndcg@3 : %.4f , ndcg@5 : %.4f " % (epoch_ndcg[0], epoch_ndcg[2], epoch_ndcg[4]))
        test_acc_k = []
        test_loss = []
        test_ndcg_k = []
        for batch_idx, test in enumerate(tqdm(test_loader)):
            print("Batch Indec:", batch_idx)
            opt.zero_grad()
            x_index_list, y_index_list = test[0], test[1]
            x_embedding_list = []
            y_embedding_list = []
            y_one_hot_list_batch = []
            for i in range(len(x_index_list)):
                index = x_index_list[i]
                index = index / 2
                index = int(index)
                x_words = sentence_data_test[index]
                y_words = label_data_test[y_index_list[i]].data.cpu()
                list_y_one_hot = []
                for i in range(len(y_words)):
                    if y_words[i] == '1':
                        list_y_one_hot.append(1)
                    else:
                        list_y_one_hot.append(0)
                y_one_hot_list_batch.append(list_y_one_hot)
                x_embedding_list.append(get_embedding_vector(x_words, 500))
                y_embedding_list.append(get_embedding_vector_label(y_words))
            x = torch.stack(x_embedding_list, 0)
            y = torch.stack(y_embedding_list, 0)
            VD = x
            VL = y
            VC = F.softmax(get_CosSim_OneByOne(VD, VL,batch_size=2),
                               dim=1)
            NVD = get_embeddings_after_NVD(VC, VD, alpha=0.1)
            NVC = F.softmax(get_CosSim_OneByOne(NVD, VL,batch_size=2),
                              dim=1)
            LA = torch.bmm(NVC, VL)
            list_NVD_test.append(NVD)
            list_LA_test.append(LA)
            y_one_hot_batch = torch.from_numpy(np.array(y_one_hot_list_batch))
            y_one_hot_batch_list_all_test.append(y_one_hot_batch)
            val_y= attention_model(NVD,LA,1,epoch_now,batch_idx)
            loss = criterion(val_y, y_one_hot_batch.float()) /test_loader.batch_size
            labels_cpu = y_one_hot_batch.data.cpu().float()
            pred_cpu = val_y.data.cpu()
            prec = precision_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            test_acc_k.append(prec)
            ndcg = Ndcg_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            test_ndcg_k.append(ndcg)
            test_loss.append(float(loss))
        avg_test_loss = np.mean(test_loss)
        test_prec = np.array(test_acc_k).mean(axis=0)
        test_ndcg = np.array(test_ndcg_k).mean(axis=0)
        print("epoch %2d test end : avg_loss = %.4f" % (epoch_now, avg_test_loss))
        print("precision@1 : %.4f , precision@3 : %.4f , precision@5 : %.4f " % (
        test_prec[0], test_prec[2], test_prec[4]))
        print("ndcg@1 : %.4f , ndcg@3 : %.4f , ndcg@5 : %.4f " % (test_ndcg[0], test_ndcg[2], test_ndcg[4]))
    for i in range(1,epochs):
        epoch_now=i+1
        print("Running EPOCH",i+1)
        train_loss = []
        prec_k = []
        ndcg_k = []
        for batch_idx, train in enumerate(tqdm(train_loader)):
            # print("-------------------------------------new batch--------------------------------")
            print("Epoch:",epoch_now)
            print("Batch Indec:",batch_idx)
            opt.zero_grad()
            NVD=list_NVD_train[batch_idx]
            LA=list_LA_train[batch_idx]
            y_one_hot_batch=y_one_hot_batch_list_all_train[batch_idx]
            y_pred= attention_model(NVD,LA,0,epoch_now,batch_idx)
            loss = criterion(y_pred, y_one_hot_batch.float())/train_loader.batch_size
            loss.requires_grad_(True)
            loss.backward(retain_graph=True)
            opt.step()
            labels_cpu = y_one_hot_batch.data.cpu().float()
            pred_cpu = y_pred.data.cpu()
            prec = precision_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            prec_k.append(prec)
            ndcg = Ndcg_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            ndcg_k.append(ndcg)
            train_loss.append(float(loss))
        avg_loss = np.mean(train_loss)
        epoch_prec = np.array(prec_k).mean(axis=0)
        epoch_ndcg = np.array(ndcg_k).mean(axis=0)
        print("epoch %2d train end : avg_loss = %.4f" % (epoch_now, avg_loss))
        print("precision@1 : %.4f , precision@3 : %.4f , precision@5 : %.4f " % (epoch_prec[0], epoch_prec[2], epoch_prec[4]))
        print("ndcg@1 : %.4f , ndcg@3 : %.4f , ndcg@5 : %.4f " % (epoch_ndcg[0], epoch_ndcg[2], epoch_ndcg[4]))
        test_acc_k = []
        test_loss = []
        test_ndcg_k = []
        for batch_idx, test in enumerate(tqdm(test_loader)):
            print("Batch Indec:", batch_idx)
            opt.zero_grad()
            x=list_NVD_test[batch_idx]
            y=list_LA_test[batch_idx]
            y_one_hot_batch=y_one_hot_batch_list_all_test[batch_idx]
            val_y= attention_model(x, y,1,epoch_now,batch_idx)
            loss = criterion(val_y, y_one_hot_batch.float()) /test_loader.batch_size
            labels_cpu = y_one_hot_batch.data.cpu().float()
            pred_cpu = val_y.data.cpu()
            prec = precision_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            test_acc_k.append(prec)
            ndcg = Ndcg_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            test_ndcg_k.append(ndcg)
            test_loss.append(float(loss))
        avg_test_loss = np.mean(test_loss)
        test_prec = np.array(test_acc_k).mean(axis=0)
        test_ndcg = np.array(test_ndcg_k).mean(axis=0)
        print("epoch %2d test end : avg_loss = %.4f" % (epoch_now, avg_test_loss))
        print("precision@1 : %.4f , precision@3 : %.4f , precision@5 : %.4f " % (
        test_prec[0], test_prec[2], test_prec[4]))
        print("ndcg@1 : %.4f , ndcg@3 : %.4f , ndcg@5 : %.4f " % (test_ndcg[0], test_ndcg[2], test_ndcg[4]))

def precision_k(true_mat, score_mat, k):
    p = np.zeros((k, 1))
    rank_mat = np.argsort(score_mat)
    backup = np.copy(score_mat)
    for k in range(k):
        score_mat = np.copy(backup)
        for i in range(rank_mat.shape[0]):
            score_mat[i][rank_mat[i, :-(k + 1)]] = 0
        score_mat = np.ceil(score_mat)
        mat = np.multiply(score_mat, true_mat)
        num = np.sum(mat, axis=1)
        p[k] = np.mean(num / (k + 1))
    return np.around(p, decimals=4)

def Ndcg_k(true_mat, score_mat, k):
    res = np.zeros((k, 1))
    rank_mat = np.argsort(score_mat)
    backup = np.copy(score_mat)
    label_count = np.sum(true_mat, axis=1)
    for m in range(k):
        y_mat = np.copy(true_mat)
        for i in range(rank_mat.shape[0]):
            y_mat[i][rank_mat[i, :-(m + 1)]] = 0
            for j in range(m + 1):
                y_mat[i][rank_mat[i, -(j + 1)]] /= np.log(j + 1 + 1)
        dcg = np.sum(y_mat, axis=1)
        factor = get_factor(label_count, m + 1)
        ndcg = np.mean(dcg / factor)
        res[m] = ndcg
    return np.around(res, decimals=4)

def get_factor(label_count,k):
    res=[]
    for i in range(len(label_count)):
        n=int(min(label_count[i],k))
        f=0.0
        for j in range(1,n+1):
            f+=1/np.log(j+1)
        res.append(f)
    return np.array(res)