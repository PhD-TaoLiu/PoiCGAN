

## Requirement

```
Python >= 3.7
pytorch >= 1.10.1
numpy >= 1.21.6
```

## Datasets

InsPLAD 可在https://drive.google.com/drive/folders/1psHiRyl7501YolnCcB8k55rTuAUcR9Ak下载。

## How to run

- 生成中毒图像：


```
python gan/dcgan.py --dataset InsPLAD
```

- 可选参数在utils/options.py
- 运行实验：

```
python main.py --dataset InsPLAD --model InsPLAD_Resnet18
```

