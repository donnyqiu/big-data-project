## 项目说明
vit-b-16在cifar-100上进行微调

## 项目结构
- logs 存放训练+推理日志 可以看时间数据
- plot 存放损失和准确率图像
- gpu_usage 存放GPU利用

## 硬件配置
- GPU: 7*RTX 4090
- Memory: 128GB
- OS: ubuntu 20.04
- CPU: Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz

## 方法
- local 单机单卡训练 即非分布式训练
    - 源代码：train.py
    - 配置：单进程 1张卡
- data_parallel_dp 单机多卡训练 数据并行
    - 方法：pytorch DataParallel 简称 DP
    - 配置：单进程利用所有7张卡
    - 源代码：train.py
- data_parallel_ddp “多机多卡”训练 数据并行
    - 方法：pytorch DistributedDataParallel 简称 DDP (all-reduce的collective架构 + nccl)
    - 配置：7进程代表7节点 每个节点拥有一张卡
    - 源代码：train.py
    - 其中“多机多卡”其实就是用多进程模拟 每个进程一张卡 底层还是使用nccl进行通信
    - 实验中只有多进程模拟的训练结果
- model_parallel 单机多卡 模型并行 拆分模型 详细拆分可见train,py
    - 模型大致拆分：cuda0:卷积层 cuda1~2:每张卡3个encoder block cuda3~5:每张卡2个encoder block cuda7:head+loss
    - 配置：单进程 利用所有7张卡
- parameter_server “多机多卡”训练 数据并行
    - 方法：parameter server模式 利用pytorch rpc通信框架实现
    - 配置：8进程代表8节点 1*ps节点+7*worker节点 每个worker节点拥有一张卡
    - 源代码：parameter_server.py


## 核心指标
- 训练时间：每个Epoch的训练时间