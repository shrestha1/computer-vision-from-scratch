import os
import pandas as pd


def get_classes()-> dict:
    file_path ='./data/tiny-imagenet-200/words.txt'
    classes = {}
    with open(file=file_path) as file:
        lines = file.readlines()
        for line in lines:
            k, v = line.split('\t')
            classes[k] = v.replace('\n', '')

    return classes

def valid_classes(train_path)->dict:
       
    all_train_classes = os.listdir(train_path)
    all_classes = get_classes()

    valid_classes = {c:all_classes[c] for c in all_train_classes if not c.endswith('.txt')}   
    
    return valid_classes



if __name__ == '__main__':
    path = './data/tiny-imagenet-200/train'
    print(valid_classes(train_path=path))