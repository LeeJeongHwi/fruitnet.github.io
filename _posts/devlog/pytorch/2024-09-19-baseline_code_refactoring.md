---
layout: post
title: Hugging Face Model을 pytorch로 학습시키는 Baseline 코드
description: >
  Boostcamp-level1 Project Baseline code
sitemap: false
categories: [devlog, pytorch]
hide_last_modified: true
---

# Hugging Face Model을 pytorch로 학습시키는 Baseline 코드

level1 첫 프로젝트를 시작하면서 받은 baseline 코드는 Pytorch Lightning으로 구현된 코드였다.

기존에 준 코드만으로 사용하는 것도 좋긴하지만, 뭔가 내 입맛대로 바꾸고싶어서 Pytorch 기반으로 만들어보기로 했다.

우선 train.py에 몰려있던 코드들을 분리시키는 것을 목적으로 했다.

* dataloader.py
    * Preprocessing, Tokernizing, get DataLoader
* dataset.py
    * return Torch dataset
* Trainer.py
    * Train, Valid, Test 등 전반적인 학습에 관한 클래스

dataloader, dataset 코드는 변경하지 않아도 될 것 같아서 따로 건드리지 않았다.

## Argparse 대신 config.yaml 사용

argparse라는 좋은 방법도 있지만, 좀 더 버전관리를 쉽게하기 위해서 config.yaml을 사용하기로 했다.

그리고 사용하는 모델자체들도 워낙 다양하기 때문에, 매번 다른 곳에 저장하는 것 보다는 yaml파일로 저장해서 config를 불러오는 것이 더 간편할 것으로 생각되어 바꾸었다.

```python
import yaml
from box import Box

def load_config(config_file):
    # Load Config.yaml
    with open(config_file) as file:
        config = yaml.safe_load(file)
        config = Box(config)

    return config
```

```yaml
# Base_config.yaml

model_name': klue/roberta-small
training:
  batch_size: 32
  max_epoch: 1
  shuffle: True
  learning_rate: 0.00002
  train_path': '../../data/train.csv'
  loss: MSELoss
  optimization:
    name: AdamW
    weight_decay: 0.01
  scheduler:
    name: ReduceLROnPlateau
    patience: 5
test:
  dev_path': '../../data/dev.csv'
  test_path': '../../data/dev.csv'
  predict_path': '../../data/test.csv'
```

```python
# train.py
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # config 파일 콘솔로 입력받기
    parser.add_argument("--config", default="Base_config", required=True)
    args = parser.parse_args()

    config_path = f"./config/{args.config}.yaml"
    config = load_config(config_path) # config 파일 불러오기
    
    dataLoader = TextDataloader(
                    model_name=config.model_name,
                    batch_size=config.training.batch_size,
                    shuffle=config.training.shuffle,
                    train_path=config.training.train_path,
                    dev_path=config.test.dev_path,
                    test_path=config.test.test_path,
                    predict_path=config.test.predict_path
                )
    dataLoader.setup(stage="fit") # DataLoader Setup
    train_loader = dataLoader.train_dataloader()
    val_loader = dataLoader.val_dataloader()
    
    trainer = torch_Trainer(config)
    trainer.train(train_loader, val_loader)
```
위 코드처럼 config를 입력 받아서 `DataLoader`나 아래에서 설명 드릴 `Trainer.py`의 인자로 사용합니다.

여기서 `--config` 는 Config File 이름을 입력 인자로 줍니다.

> 예시
> 
> `torch_code# python3 train.py --config kf_deberta_base_config`

`train_loader`와 `val_loader`를 `DataLoader`에서 받아옵니다.

```python
# Dataloader.py
def train_dataloader(self):
    return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

def val_dataloader(self):
    return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

def test_dataloader(self):
    return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

def predict_dataloader(self):
    return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)
```

## Trainer.py

Main 코드를 최대한 줄이고자 학습, 평가, 테스트를 해당 모듈로 작성하였습니다

Lightning과 달리 torch는 model, loss, optimizer를 다 설정해주어야합니다.

아직 제대로 나누지 못했지만, 각 구성요소를 `get`으로 함수화하였습니다.

```python
# Trainer.py
def get_model(self, model_name):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_name, num_labels=1
    )
    return model
        
def get_optimizer(self, model, optimizer):
    if optimizer.name == "AdamW":
        optim = torch.optim.AdamW(model.parameters(), weight_decay=optimizer.weight_decay,
                                  lr=self.lr)
    return optim

def get_loss(self, loss):
    if loss == "MSELoss":
        return torch.nn.MSELoss()
    elif loss == "l1Loss":
        return torch.nn.L1Loss()
    elif loss == "HuberLoss":
        return torch.nn.HuberLoss()
    # Add Loss
```

각 구성 요소들을 입력 받은 Config를 기반으로 반환합니다.

> 추후에 모델을 더 많이 사용한다고 하면, 따로 `py` 파일을 만들어서 저장하는 방향이 좋습니다.
> 
> (KR-SBERT의 경우 따로 만들어야함)


다음은 학습 과정입니다.

```python
def train(self, train_loader, val_loader):
    # Set initial
    model = self.get_model(self.model_name) 
    optim = self.get_optimizer(model=model, optimizer=self.optimizer)
    criterion = self.get_loss(self.loss)
    lr_scheduler = self.get_scheduler(optim, self.scheduler, verbose=True)
    model.to(self.device)
    best_pearson = 0.0
    
    # model train 
    model.train()
    for epoch in range(self.epoch):
        train_bar = tqdm(train_loader)
        for step, batch in enumerate(train_bar):
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Calc Loss
            outputs = model(x)
            loss = criterion(outputs.logits.squeeze(), y.squeeze())
            
            # update weights and Scheduler
            loss.backward()
            optim.step()
            optim.zero_grad()
            # lr_scheduler.step()
            train_bar.desc=f"Train Epoch[{epoch+1}/{self.epoch}] loss : {loss}"

        # Epoch별 Validation
        pearson = self.valid(model, criterion, val_loader)
    
        # validation Pearson에 따라 Ckpt 저장
        if pearson > best_pearson: # Best Pearson 저장
            ckpt_save(model, self.model_name, optim, self.epoch, pearson, best_pearson)
            best_pearson = pearson
```

`outputs = model(x)`

batch 단위로 `train_loader`에서 데이터를 받아와 GPU할당 후 `x`를 model에 전달해 예측 결과를 받아옵니다.

`loss = criterion(outputs.logits.squeeze(), y.squeeze())`

그리고 위에서 설정한 `loss`로 `label(y)` 와 예측 값에 대한 손실 값을 구합니다.

`loss.backward(); optim.step(); optim.zero_grad()`

구한 손실 값을 기반으로 가중치를 업데이트하고, 이를 반복합니다.

`pearson = self.valid(model, criterion, val_loader)`

Epoch마다 validation을 수행해서 valid dataset에 대한 Pearson 계수를 확인합니다. (아래에 valid 코드 설명 예정) 그 후 이전 Pearson 보다 높을 시 `best_pearson`으로 교체하고, 해당 에폭에 대한 모델을 저장합니다. (checkpoint 개념)


```python
def valid(self, model, criterion, val_loader):
    model.eval()
    val_bar = tqdm(val_loader)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for step, batch in enumerate(val_bar):
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            outputs = model(x) # output) outputs.logits (예측결과)
            loss_v = criterion(outputs.logits.squeeze(), y.squeeze()) # validation Loss라서 없어도 됨

            # Batch 별로 예측한 데이터와 label값들을 전체 데이터로 넣어줌
            all_preds.append(outputs.logits.squeeze())
            all_labels.append(y.squeeze())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    pearson = torchmetrics.functional.pearson_corrcoef(all_preds, all_labels) # Pearson 상관계수
    print("======================================================")
    print(f"            Pearson Coeff : {pearson:.4f}")
    print("======================================================")
    return pearson
```

위 train과 방식은 동일하나, batch단위가 아닌 valid dataset 전체 개수에 대한 Pearson 을 계산해야하기 때문에, `all_preds, all_labels` 리스트를 추가하였습니다.

배치마다 예측한 값들을 하나의 리스트로 만들기 위해 `outputs.logits.squeeze()`를 하여 shape을 `[16]` 으로 맞추어 append 하였습니다.

예시로, 데이터가 320이고, 배치가 32일 때, 총 10번의 step이 나옵니다. 이때의 `all_preds`와 `all_labels`의 shape은 다음과 같습니다.

`[[3.1423, 2.1231, 1.2321, ....], [1.231, ...], [2.3123, ...] ,[4.3123, ...] , [....]], shape=[10,32]`

그리고 이를 `concat`하여 `[320]`의 shape를 만들어주어 전체 데이터에 대한 예측 값과 레이블 값을 만들어 냅니다.

그 후 `torch.metrics.functional.pearson_corrcoef(all_preds, all_labels)`하여 Pearson을 계산합니다.

`Trainer.py` 내에서는 다음과 같이 연산을 하고, 실제 main 코드에서는 두 줄로 사용됩니다.

```python
trainer = torch_Trainer(config)
trainer.train(train_loader, val_loader)
```

Model Save는 다음과 같이 진행됩니다

```python
def ckpt_save(model, model_name, optimizer, epoch, pearson, best_pearson):

    model_directory = "./saved_model"
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    
    save_path = f"{model_name.split('/')[1]}_best_model_Pearson_{pearson}_epoch_{epoch}.pt"
    torch.save(model, os.path.join(model_directory,save_path))
    
    print(f"Model Saved at Pearson {best_pearson} to {pearson}")
```

`best_pearson`이 갱신될 때마다 model을 저장하는 코드입니다.

실행되는 `train.py` 을 기준으로 saved_model 폴더를 생성하고 다음과 같이 저장합니다.

`kf_deberta_base_best_model_pearson_{Pearson}_epoch_{현재epoch}.pt`

이 모델은 아래 설명드릴 `inference.py`에서 불러와 추론 과정을 거칩니다.

## inference.py

이전 `train.py`와 거의 동일하지만, AIstage에서 제공한 prediction과 output.csv를 추출하는 코드를 사용하여 리더보드용 결과파일을 추출합니다.

```python
# inference.py
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Base_config", required=True)
    parser.add_argument("--saved_model", required=True)
    args = parser.parse_args()

    ...
    predict_loader = dataLoader.predict_dataloader()
    test_loader = dataLoader.test_dataloader()
    
    trainer = torch_Trainer(config)
    model = torch.load(f"./saved_model/{args.saved_model}.pt")
    predictions = trainer.predict(model=model, dataloader=predict_loader)
    predictions = list(round(float(i), 1) for i in predictions)

    output = pd.read_csv("../../data/sample_submission.csv")
    output["target"] = predictions
    output.to_csv('./output/output.csv', index=False)
    print("Complete Extract ouptut.csv")
```

`--config, --saved_model`인자에 다음과 같이 입력으로 주어 실행합니다.

> `torch_code/# python3 inference.py --config kf_deberta_base --saved_model kf_deberta_base_best_model_pearson_{Pearson}_ecpoh_{epoch}`

(해당 모델 저장 방식은 조금 길다고 생각하여 추후에 변경하는 것이 좋아보입니다)

```python
# Trainer.py
def predict(self, model, dataloader):
    model.eval()
    all_preds = []
    with torch.no_grad():
        predict_bar = tqdm(dataloader)
        for step, batch in enumerate(predict_bar):
            x = batch
            x = x.to(self.device)
            predict = model(x)
            
            all_preds.append(predict.logits.squeeze())
    
    predictions = torch.cat(all_preds)
    
    return predictions
```

`predict_loader`에는 `y`(label)값이 존재하지 않습니다 (리더보드용 데이터셋)

따라서 `x`에 대해서 예측하고 그에 대한 값을 `predictions`로 반환합니다.

그 후 `predictions = list(round(float(i), 1) for i in predictions)` 코드로 n자리 소수점을 1번째 자리까지 반올림하여 `1.5` 와 같이 만들어 output.csv에 저장합니다.

`kf-deberta-base` 모델에 대해서 예시로 해보았을 때, 리더보드에도 잘 올라갑니다.

이렇게 최종적으로 모든 과정이 마무리가 됩니다.

## 개선이 필요한 사항들

- Scheduler 구현
    
- Wandb 구현
    
- Model, Optimizer 등등 구성요소 모듈화
    
    - 만약에 Model별 학습방법이 다르다면 어떻게 해야하는가.. Trainer를 따로 만들어야하는가..? (KR-SBERT 예시)

