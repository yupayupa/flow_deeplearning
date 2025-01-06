import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn

import lrp
from .functional import linear

from torcheval.metrics.functional import multiclass_accuracy
from torcheval.metrics.functional import multiclass_precision
from torcheval.metrics.functional import multiclass_recall

from early_stopping_pytorch import EarlyStopping


#ファイルの読み込み
normal_data=pd.read_csv("./csv/success-csv/result.csv")
attack_data=pd.read_csv("./csv/attack-csv-2023/attack-2023-03.csv")

#データのラベル付け
normal_data["label"]=0
attack_data["label"]=1

#正規データ、攻撃データの結合
data = pd.concat([normal_data, attack_data]).reset_index(drop=True)


#特定の列を削除
data= data.drop(
        [
            "src_ip",
            "dst_ip",
            "src_port",
            "dst_port",
            "src_mac",
            "dst_mac",
            "protocol",
            "timestamp"
        ],
        axis=1,
        )



#空データの処理
data = data.fillna(0)

X = data.drop(columns=["label"]).values.astype(np.float32)
y = data["label"].values.astype(np.float32)

scaler = StandardScaler()
X = scaler.fit_transform(X)


class NetworkDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]





#データをラベルとデータ、テストと訓練の要素により4つに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#データセットを訓練データとテストデータにまとめる
train_dataset = NetworkDataset(X_train, y_train)
val_dataset = NetworkDataset(X_val, y_val)
test_dataset = NetworkDataset(X_test, y_test)

#バッチの生成とデータのシャッフルを行う
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


#ニューラルネットワークの構造を決定する
class NeuralNetwork(nn.Module):
    def __init__(self,input_size):
        super(NeuralNetwork,self).__init__()
        self.linear_relu_stack=nn.Sequential(
                lrp.Linear(input_size,256),
                nn.ReLU(),
                lrp.Linear(256,64),
                nn.ReLU(),
                lrp.Linear(64,2)
                
                )

    
    def forward(self,x):
        return self.linear_relu_stack(x)


input_size=X_train.shape[1]
model= NeuralNetwork(input_size)

#交差エントロピー誤差とadamを使用する
criterion=nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-4)
epoch=0

early_stopping = EarlyStopping(patience=7,verbose=True)

#訓練開始
for i in range(100):
    model.train()
    total_loss=0.0  

    for batch_idx, (features, labels) in enumerate(train_loader):
        optimizer.zero_grad() #勾配のリセット
        outputs = model(features)
        loss = criterion(outputs, labels.type(torch.long))
        loss.backward() #誤差逆伝播
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_features, val_labels in val_loader:
            val_outputs = model(val_features)
            val_loss += criterion(val_outputs, val_labels.type(torch.long)).item()

    early_stopping(val_loss, model)

    if early_stopping.early_stop:
        print("Early Stop")
        break

    epoch+=1


    print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader):.4f}")


from sklearn.metrics import accuracy_score # 正解率
from sklearn.metrics import precision_score # 適合率
from sklearn.metrics import recall_score # 再現率
from sklearn.metrics import f1_score # F1値


#評価
model.eval()  # これによってDropout等をなくす
#all_predictions = []
#all_labels = []

total=0
correct=0

#for features, labels in test_loader:
#    outputs = model(features)
#    predictions = torch.argmax(outputs, dim=1)
#    all_predictions.append(predictions)
#    all_labels.append(labels.type(torch.long))

#for features, labels in test_loader:
#    outputs = model(features)
#    predicted=torch.argmax(outputs,dim=1)




# 精度、適合率、再現率
#all_predictions = torch.cat(all_predictions)
#all_labels = torch.cat(all_labels)

#accuracy = multiclass_accuracy(all_predictions, all_labels, num_classes=2)
#precision = multiclass_precision(all_predictions, all_labels, num_classes=2)
#recall = multiclass_recall(all_predictions, all_labels, num_classes=2) 

#print(f"Test Accuracy: {accuracy.item():.20f}")
#print(f"Test Precision: {precision.item():.20f}")
#print(f"Test Recall: {recall.item():.20f}")
from sklearn.metrics import accuracy_score # 正解率
from sklearn.metrics import precision_score # 適合率
from sklearn.metrics import recall_score # 再現率
from sklearn.metrics import f1_score # F1値

from torch.autograd import Variable

X_test = torch.Tensor(X_test)

outputs = model(Variable(X_test))
predicted = torch.argmax(outputs.data, 1)

print(f"正解率 :{accuracy_score(y_test, predicted)}")
print(f"適合率 :{precision_score(y_test, predicted)}")
print(f"再現率 :{recall_score(y_test, predicted)}")
print(f"F1スコア :{f1_score(y_test, predicted)}")

