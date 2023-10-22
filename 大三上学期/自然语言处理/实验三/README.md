## 一. 实验名称

​	故事生成



## 二. 实验内容

​	利用**RNN系列模型，GPT系列模型等完成故事生成**。



## 三. 实验启发

- 可以**以字符为单位进行文本生成**，也可以**以单词为单位进行文本生成**，二者生成效果不同

- **构造训练数据有多种方法**，不同方法训练得到的模型生成效果存在差异

- 训练数据的**sequence length会影响生成效果**

- 从**输出分布中抽样**可以避免模型重复生成相同内容

- **温度参数**会影响生成效果

- **使用Hugging Face管道应用NLP预训练模型** (以GPT2为例)

  - 搭建GPT2

    ```python
    # GPT2LMHeadModel主体为调用GPT2Model类以及一个输出层self.lm_head
    # self.lm_head将GPT2Model类最后一个Block输出的hidden_states张量的最后一个维度由768维(config.n_embd)投影为词典大小维度(config.vocab_size)的lm_logits张量
    model = GPT2LMHeadModel.from_pretrained("./gpt2")
    ```

  - 构造GPT2输入

    ```python
    # 构造分词器
    tokenizer = GPT2Tokenizer.from_pretrained("./gpt2")
    # GPT2模型的target_sequence与input_sequence相同
    input_sequence = tokenizer.encode(text)
    ```

  - 微调GPT2

    ```python
    # 微调GPT2模型时不需要再定义损失函数，GPT2模型已经内嵌了交叉熵损失
    model.train()
    outputs = model(input_sequences, labels = target_sequences)
    loss = outputs[0]
    ```

  - GPT2生成

    ```python
    model.eval()
    with torch.no_grad():
         outputs = model(sequence_tensor)
         pred = outputs[0]
    pred = pred[0, -1, :] / temperature
    # GPT2采用了top-k机制，模型会从概率前k大的token中抽样选取下一个token
    pred_index = select_top_k(pred, k=10)
    pred_text = tokenizer.decode([pred_index])
    ```

    