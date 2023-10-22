## 一. 实验名称

​	基于多种编码器的文本理解



## 二. 实验内容

​	利用**Bi-LSTM，Transformer，Bert等模型完成文本分类**。



## 三. 实验启发

- 对**预训练模型微调**时，**学习率和初始参数应该设置地小一些**，避免预训练模型不收敛

- **调用预训练词向量实现词嵌入**的两种方式

  ```python
  ## 方式一 ##
      # vocab_size：词汇表大小
      # ninp：词嵌入维度
      # embedding_weight：预训练词向量
      self.embed = nn.Embedding(vocab_size, ninp)
      self.embed.weight.data.copy_(embedding_weight)
      self.embed.weight.requires_grad = True # 是否进行Freeze
          
      ## 方式二 ##
      # embedding_weight：预训练词向量
      self.embed = nn.Embedding.from_pretrained(embedding_weight, freeze=False)
  ```

- **使用Hugging Face管道应用NLP预训练模型** (以Bert为例)

  - 搭建Bert模块

    ```python
    # BERT_PATH：从Hugging Face下载下来的预训练模型参数存储路径，也可以用预训练模型名称替换，进行在线下载
    self.bert_config = BertConfig.from_pretrained('bert-base-uncased')
    self.bert = BertModel.from_pretrained(BERT_PATH, config = self.bert_config)
    ```

  - Bert模块输入输出

    ```python
    # input_ids：token id
    # attention_mask：区分哪些token是padding的 避免模型attention到padding token
    # token_type_ids：区分不同token sequence
    # bert_output[0] 包含 sequence 中所有 token 的 embedding 向量
    # bert_output[1] 包含 [CLS] token 的 embedding 向量
    # self.bert 的输出类型是 BaseModelOutputWithPoolingAndCrossAttentions
    bert_output = self.bert(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
    ```

  - 构造Bert模块输入

    ```python
    # 构造分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # text：待处理文本
    # add_special_tokens：是否添加[CLS]等special token
    # padding：是否padding待处理文本
    # truncation：当待处理文本长度超过阈值时，是否进行截断处理
    # max_length：待处理文本长度阈值
    token = tokenizer(text, add_special_tokens = True, padding = 'max_length', truncation = True, max_length = 150)
    input_ids = token['input_ids']
    token_type_ids = token['token_type_ids']
    attention_mask = token['attention_mask']
    ```

- 电脑资源不够的话，AutoDL、腾讯云等是时候登场了🌝



## 四. 实验参考资料

- LSTM 反向传播公式推导

  [LSTM参数更新推导 | 记录思考 (ilewseu.github.io)](https://ilewseu.github.io/2018/01/06/LSTM参数更新推导/)

- Transformer Positional Encoding 原理与推导

  [一文教你彻底理解Transformer中Positional Encoding - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/338592312)

  [深度学习入门--Transformer中的Positional Encoding详解_class positionalencoding(nn.module):_CuddleSabe的博客-CSDN博客](https://blog.csdn.net/qq_15534667/article/details/116140592)

- 加载模型预训练参数

  [PyTorch：如何加载预训练参数？_torch.load载入训练参数-CSDN博客](https://blog.csdn.net/fhcfhc1112/article/details/95862915)

  
