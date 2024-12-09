## 项目结构
- logs 存放训练日志 有时间数据
- plot 存放损失和准确率图像

## 方法
- local 单机单卡训练
- data_parallel 单机多卡+“多机多卡”训练 数据并行
- model_parallel 单机多卡 模型并行


### 多机多卡环境模拟：
1机7卡模拟2机7卡
只在数据并行上做

- 节点1：4GPU
- 节点2：3GPU

使用数据并行 每个GPU上部署一个模型

使用torchrun模拟
- 模拟第一台机器
```
NODE_RANK=0 NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 main.py --master_addr="127.0.0.1" --master_port=12355
```

- 模拟第二台机器
```
NODE_RANK=1 NPROC_PER_NODE=3 CUDA_VISIBLE_DEVICES=4,5,6 torchrun --nnodes=2 --node_rank=1 --nproc_per_node=3 main.py --master_addr="127.0.0.1" --master_port=12355
```

效果和单机多卡没有本质区别 直接使用data_parallel单机多卡的logs和plot即可

### 待实验
缓存组件的使用