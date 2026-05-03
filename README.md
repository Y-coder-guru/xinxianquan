# 新鲜圈 — 分类预测项目

## 目录结构

```
xinxianquan/
├── 源码/                  # 源代码
│   └── train_and_predict.py
├── 模型/                  # 训练好的模型文件（运行后生成）
│   └── model.joblib
├── 提交结果/              # 预测结果文件（运行后生成）
│   └── submission.csv
├── docker容器/            # 容器化相关文件
│   ├── Dockerfile
│   └── docker-compose.yml
├── train_data.csv         # 训练数据（含 id、label 及特征列）
├── test_data.csv          # 测试数据（含 id 及特征列）
├── sample_submission.csv  # 提交格式示例
├── requirements.txt       # Python 依赖列表
└── README.md
```

## 环境要求

- Python 3.11 或更高版本

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行训练与预测

在仓库根目录下执行：

```bash
python 源码/train_and_predict.py
```

运行完成后：

- 训练好的模型保存至 `模型/model.joblib`
- 预测结果保存至 `提交结果/submission.csv`

## 训练策略说明

- 脚本会先输出标签分布以及 5 折交叉验证的基线准确率（HistGradientBoostingClassifier，使用相同的缺失值处理与标准化以便对比）。
- 使用 `SimpleImputer` + `RobustScaler` 进行缺失值处理与特征标准化。
- 采用 `ExtraTreesClassifier` 并通过 `RandomizedSearchCV`（3 折）进行简洁的超参搜索。
- RandomizedSearchCV 默认会在全量数据上重新拟合最佳模型，脚本直接使用该模型生成提交文件。
- 可在 `源码/train_and_predict.py` 中调整 `CV_FOLDS`、`SEARCH_ITER` 与 `EXTRATREES_PARAM_SPACE`。

## 使用 Docker 运行

### 构建并运行容器

```bash
docker compose -f docker容器/docker-compose.yml up --build
```

运行完成后，`模型/` 和 `提交结果/` 目录会通过卷挂载同步到宿主机。

### 单独使用 Dockerfile

```bash
# 在仓库根目录构建镜像
docker build -f docker容器/Dockerfile -t xinxianquan .

# 运行容器并挂载输出目录
docker run --rm \
  -v "$(pwd)/模型:/app/模型" \
  -v "$(pwd)/提交结果:/app/提交结果" \
  xinxianquan
```

## 输出格式

`提交结果/submission.csv` 格式与 `sample_submission.csv` 一致，包含两列：

| id  | label   |
|-----|---------|
| 0   | class_X |
| 1   | class_Y |
| ... | ...     |
