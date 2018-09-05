# TaskBot

TaskBot是一个基于深度学习、机器学习、规则系统的任务驱动型多轮对话框架，欢迎star、contribute。100行代码，实现你的chatbot。

## 特性

+ 多轮跨场景语义继承
+ 模块化组件，面向快速的2次定制化开发任务
+ 微信接口、Http接口支持

## TODO

### 文本处理与表示

- [ ] 分词
- [ ] 文本向量化
  - [ ] Tfidf
  - [ ] word2vec
  - [ ] doc2vec
  - [ ] LDA

### 自然语言理解（NLU）
- [ ] 命名实体识别
  - [ ] Keywords Match
  - [ ] CRF
  - [ ] BiLSTM + CRF
- [ ] 命名实体归一
  - [ ] Fuzzy Match
  - [ ] Rule Base
- [ ] 意图识别
  - [ ] 单轮语境意图识别
    - [ ] TextCNN
    - [ ] FastText
    - [ ] Keywords Match
  - [ ] 多轮语境意图识别
    - [ ] LSTM
    - [ ] Keywords Match
    - [ ] Rule Base
- [ ] 情感识别

### 会话状态追踪（DST）

包括上下文、实体、意图、回复、情感等语境的记录、使用、向量化表示。
- [ ] 会话追踪
  - [ ] 上下文
  - [ ] 实体
  - [ ] 意图
  - [ ] 回复
  - [ ] 情感
  - [ ] 向量化表示

- [ ] 会话追踪序列化
  - [ ] 保存/更新
  - [ ] 数据集转换

### 回复

- [ ] 基于检索
  - [ ] 相似度排序
    - [ ] tfidf similarity
    - [ ] word2vec/doc2vec similarity
    - [ ] keywords match score
  - [ ] 条件筛选
    - [ ] rule base
    - [ ] database
- [ ] 逻辑推断
  - [ ] 知识图谱（knowledge base）
- [ ] 基于生成
  - [ ] seq2seq
- [ ] 基于规则
 - [ ] 模式匹配
### 数据集

- [ ] 语料数据结构、统一格式定义
- [ ] 语料处理类

### 服务接口
- [ ] 微信
- [ ] HTTP
- [ ] terminal


## LICENSE

TaskBot is licensed under the [ GNU GENERAL PUBLIC LICENSE VERSION 3.0](./LICENSE)

