# MolCut

分子化学键切割位点预测工具，用于逆合成分析。

## 项目结构

```
molcut-2/
├── disconnect_code/     # 深度学习模型训练代码
├── mapper/              # 化学键切割数据处理
├── spider/              # 有机化学命名反应爬虫
└── rdeditor-disconnect/ # RDKit 分子编辑器
```

## 模块说明

- **spider**: 从 organic-chemistry.org 爬取命名反应数据
- **disconnect_code**: 基于深度学习的分子断键预测模型
- **mapper**: 化学键切割数据处理与转换
- **rdeditor-disconnect**: 分子结构可视化编辑器

## 依赖

```bash
pip install torch rdkit pandas numpy
```

## 使用方法

详见各子目录下的 README.md 文件。
