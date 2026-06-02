# Langchain进阶1

**本节主要包括1.Model I/O,2.Ollama本地大模型部署,3.提示词PromptTemplate和模型调用方法**

## Model I/O (大模型接口)

Model I/O就是一套标准的大模型输入输出的模板

三件套就是:**输入提示(Format),调用模型(Predict),输出解析(Parse)**就是输入,处理,输出三个部分

我们主要使用的是Chatmodel输入的是一个消息列表,返回的是一个对话对象(AIMessage)

调用大模型的时候

如果没有具体规范使用openai

```python
from langchain_openai import ChatOpenAI
import os

chatLLM = ChatOpenAI(
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",  # 此处以qwen-plus为例，您可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    # other params...
)
```

有具体规范去具体官网查找,如deepseek

```python
from langchain_deepseek import ChatDeepSeek
model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("deepseek-api"),
)
```

这两个基本相同openai里也有deepseek,而qianwen需要下载软件包再导入(官方文档中可见)

```python
#pip install langchain-community
#pip install dashscope
```

下载软件包后导入

```python
import os
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage

chatLLM = ChatTongyi(
    model="qwen-plus",
    api_key=os.getenv("aliQwen-api"),
    streaming=True,
    model_provider="openai"
    # other params...
)
```

即可.

练习:导入zhipuAI

![image-20260516092753688](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260516092753688.png)

```python
from langchain_community.chat_models import ChatZhipuAI
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv(override=True)

chat = ChatZhipuAI(
    model="glm-4",
    temperature=0.5,
)
messages = [
    SystemMessage(content="你是一名专业的翻译家，可以将用户的中文翻译为英文。"),
    HumanMessage(content="我喜欢编程。"),
]
result = chat.invoke(messages)
print(result.content)
```

![image-20260516094237274](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260516094237274.png)

## Ollama本地大模型部署

在本地下载ollama来本地部署大模型

主要不要下载在c盘

为了私密化数据

在下载ollama中调出控制台

![image-20260516101710990](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260516101710990.png)

ollama run "模型名"

下载模型

在python中使用本地模型

先安装软件包

pip install -qU langchain-ollama

pip install -U ollama

```python
from langchain_ollama import ChatOllama
model= ChatOllama(base_url="http://127.0.0.1:11434", model="qwen:4b",reasoning= False)
print(model.invoke("你能干什么").content)
```

![image-20260516101853726](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260516101853726.png)

注意文件名不要冲突

## 模型调用方法

共有六种基础的调动方法

**同步异步**

![image-20260516113908166](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260516113908166.png)

**普通调用有两种同步和异步**

```#
LangChain 提供 ainvoke() 异步调用接口，用于在 异步环境（async/await） 中高效并行地执行模型推理。
它的核心作用是：让你同时调用多个模型请求而不阻塞主线程 —— 特别适合大批量请求或 Web 服务场景（如 FastAPI）
```

同步

```python
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage,SystemMessage
load_dotenv(encoding="utf-8",override= True)
model = init_chat_model(
    model="deepseek-v4-pro",
    model_provider="openai",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
messages=[
    SystemMessage(content="你是一个数学专家,不是数学问题不回答"),
    #HumanMessage(content="母猪的产后护理")
    HumanMessage(content="1+1等于几")
]
response= model.invoke(messages)
print(type( response))
print(response.content)
print(response.content_blocks)
```

异步

```python
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage,SystemMessage
load_dotenv(encoding="utf-8",override= True)
model = init_chat_model(
    model="deepseek-v4-pro",
    model_provider="openai",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
messages=[
    SystemMessage(content="你是一个数学专家,不是数学问题不回答"),
    #HumanMessage(content="母猪的产后护理")
    HumanMessage(content="1+1等于几")
]
response= model.invoke(messages)
print(type( response))
print(response.content)
print(response.content_blocks)
```



**流式调用也是同步和异步**

同步

```python
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage,SystemMessage
load_dotenv(encoding="utf-8",override= True)
model = init_chat_model(
    model="deepseek-v4-pro",
    model_provider="openai",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
messages = [
    SystemMessage(content="你叫小问，是一个乐于助人的AI人工助手"),
    HumanMessage(content="你是谁")
]
response = model.stream(messages)
print(f"响应类型：{type(response)}")
# 流式打印结果
for chunk in response:
    # 刷新缓冲区 (无换行符，缓冲区未刷新，内容可能不会立即显示)
    print(chunk.content, end="",flush=True)
print("\n")
```

异步

```python
import os
import asyncio
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
load_dotenv(encoding="utf-8",override= True)
model = init_chat_model(
    model="deepseek-v4-pro",
    model_provider="openai",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
messages=[
    SystemMessage(content="你是一个数学专家,不是数学问题不回答"),
    #HumanMessage(content="母猪的产后护理")
    HumanMessage(content="1+1等于几")
]
async def async_stream_call():
    # astream 返回异步生成器，无需 await 修饰，直接赋值
    response = model.astream(messages)
    print(f"响应类型：{type(response)}") # 响应类型：<class 'async_generator'>

    # 异步遍历异步生成器（必须使用 async for，不可用普通 for）
    # 异步遍历异步生成器（必须使用 async for，不可用普通 for）
    # 异步遍历异步生成器（必须使用 async for，不可用普通 for）
    async for chunk in response:
        # 刷新缓冲区，实现流式打印（无换行、即时输出）
        print(chunk.content, end="", flush=True)
    print("\n")

# 4.运行异步函数
if __name__ == "__main__":
    asyncio.run(async_stream_call())
```

**批处理也是同步和异步**

batch 同步输出

```python
from langchain_ollama import ChatOllama


model= ChatOllama(base_url="http://127.0.0.1:11434", model="qwen:4b",reasoning= False)
question= [
    "你是谁",
    "1+1等于几",
    "简单介绍一下langchain"
]
response= model.batch(question)
print(type(response))
for q,r in zip(question,response):
    print(f"问题{q}\n回答{r.content}\n===========================")
```

异步输出

```python
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
import asyncio
from langchain.messages import HumanMessage,SystemMessage
load_dotenv(encoding="utf-8",override= True)
model = init_chat_model(
    model="deepseek-v4-pro",
    model_provider="openai",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
questions=[
    "什么是redis?简洁回答，字数控制在100以内",
    "Python的生成器是做什么的？简洁回答，字数控制在100以内",
    "解释一下Docker和Kubernetes的关系?简洁回答，字数控制在100以内"
]
async def async_batch_call():
    response= await model.abatch(questions)
    print(type(response))
    for q,r in zip(questions,response):
        print(f"问题{q}\n回答{r.content}\n===========================")
if __name__ == "__main__":
    asyncio.run(async_batch_call())
```

### **注意!!!!**![image-20260516115729844](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260516115729844.png)

astream不需要await,因为astream返回值是一个异步生成器但是必须使用必须使用 async for遍历!!!!

在批处理时使用的时zip函数来输出

```python
 for q,r in zip(questions,response):
        print(f"问题{q}\n回答{r.content}\n===========================")
```

## 提示词PromptTemplate

#### 提示词主要分成4个部分

**设定系统SystemMessage,**

**人说的话HumanMessage,**

**AI的回复AIMessage,**

**AI使用的工具ToolMessage(v1.0)/FunctionMessage(v0.3)**

![image-20260516133123815](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260516133123815.png)

#### 提示词模板

**使用提示词模板来传递输入,更清晰地表达用户意图，更好地利用模型能力**

#### 提示词模板分类

**1.PromptTemplate：文本生成模型提示词模板，用字符串拼接变量生成提示词**

**2.ChatPromptTemplate：聊天模型提示词模板，适用于如 gpt-3.5-turbo、gpt-4 等聊天模型**

**3.FewShotPromptTemplate：(了解):少样本学习提示词模板， 构建一个Prompt其中包含多个示例，可以自动将这些示例格式化并插入到主Prompt 中形成样本提示模板，通过在给模型的最终输入中筛入一些示例，来教模型如何回答**

**4.PipelinePrompt(了解)：管道提示词模板，用于把几个提示词组合在一起使用**

#### PromptTemplate文本提示词模板

在0.3版本中使用

SystemMessage="  "

HumanMessage="  "

在1.0之后更推荐

[

("system","   "),

("human","   ")

("ai","   ")

]格式

提示词工程相关代码

```python
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage,SystemMessage
from langchain_core.prompts import PromptTemplate
load_dotenv(encoding="utf-8",override= True)
#创建模型对象
model = init_chat_model(
    model="deepseek-v4-pro",
    model_provider="openai",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=2.0
)
#创建模板
template = PromptTemplate(
    template="你是一个专业的{role}工程师，请回答我的问题给出回答，我的问题是：{question}",
    input_variables=['role', 'question']
)
#填充模板
prompt = template.format(role="Java", question="写一段简单的循环")
model1= model.invoke(prompt)
print(model1.content)


#同样的道理
template1= PromptTemplate(
    template="你是{role},你需要告诉我{question}",
    input_variables=['role', 'question']
)
prompt1= template1.format(role="数学老师", question="1+1=?")
print(type(prompt1))
result= model.invoke(prompt1)
print(result.content)
print(result.content_blocks)
```

ps:format是初始化使用
