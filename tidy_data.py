import csv


def form_data(file_path):
    sentence_list_all = []
    label_list_all = []
    num = 0
    with open(file_path, 'r') as f:
        for row in csv.reader(f):
            num = num + 1
            sentence=''
            sentence_list = []
            count = 0
            for i in row[0]:
                if i >= '0' and i <= '9':
                    count += 1
                else:
                    break
            a = row[0][0:count]
            label_list_all.append(a)
            b = row[0][count + 1:]
            sentence=sentence+b
            for i in range(1, len(row)):
                sentence=sentence+' '+row[i]
            sentence_list_all.append(sentence)
    return  sentence_list_all, label_list_all,num
