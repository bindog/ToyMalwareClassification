#微软恶意代码分类

比赛说明和数据下载 
https://www.kaggle.com/c/malware-classification/

##代码说明
- `randomsubset.py` 抽取训练子集
- `asmimage.py` ASM文件图像纹理特征
- `opcode_n-gram.py` Opcode n-gram特征
- `firstrandomforest.py` 基于ASM文件图像纹理特征的随机森林
- `secondrandomforest.py` 基于Opcode n-gram特征特征的随机森林
- `combine.py` 将两种类型的特征结合

##运行说明

1. 将完整的训练数据集解压，修改`randomsubset.py`中的路径并运行
2. 修改`asmimage.py`和`opcode_n-gram.py`中的路径，并运行`run.sh`，耐心等待即可看到结果