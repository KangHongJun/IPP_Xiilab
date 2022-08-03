#https://tutorials.pytorch.kr/beginner/basics/data_tutorial.html


#더 좋은 가독성, 모듈성을 위한 데이터 세트옵션을 미리 로드된 데이터에 대해 시본 설정 제공
#FashionMNIST - 600000개의 예시와 10000개의 테스트예시, 28*28이미지오 10개의 class

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root="data",
    train = True,
    download =True,
    transform=ToTensor(),  #이미지 데이터 [0,1]값으로 변경
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform = ToTensor(),
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize=(8,8))
cols, rows = 3,3
for i in range(1, cols*rows +1):
    sample_idx = torch.randint(len(training_data),size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows,cols,i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

#파일에서 데이터셋
#기억에 남는 패션 MNIST 이미즌 img_dir 기억에 annotations_file 남는다고 한다.

import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    #init은 dataset 객체가 생성될때마다 발생
    def __init__(self,annnotations_file,img_dir, transform=None,target_transfrom=None):
        self.img_labels = pd.read_csv(annnotations_file, names=['file_name','label'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transfrom

    #함수의 데이터셋의 샘플 개수  반환
    def __len__(self):
        return len(self.img_labels)

    #주어진 인덱스 idx에 해당하는 샘플을 데이터 셋에서 불러오고 반환
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0])
        image = read_mage(img_path) #이미지를 텐서로 변형
        label = self.img_labels.iloc[idx,1] #정답 레이블 가져옴

        #해당하면 변형하여 텐서 이미지와 라벨을 dict형으로 반환
        if self.transform:
            iamge  =self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image,label

#DataLoader로 학습용 데이터 준비
#Dataset은 데이터셋의 특징을 가져오고 하나의 샘플에 정답을 지정하는 일을 한번에 한다.
#모델을 학습할 때 일반적으로 샘플은 미니배치로 전달하고, 에폭마다 데이터를 섞어서 과적합을 막고 multiprocessing으로 속도up
#다음은 복잡한 과정을 추상화한 순회가능한 객체(iterable)

from torch.utils.data import DataLoader

train_datalodaer = DataLoader(training_data,batch_size=64, shuffle=True)
test_datalodaer = DataLoader(test_data,batch_size=64, shuffle=True)

train_features, train_labels = next(iter(train_datalodaer))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img,cmap="gray")
plt.show()
print(f"Label: {label}")


















