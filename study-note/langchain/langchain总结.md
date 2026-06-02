# 前言

为了方便自己理解新知识同时**复习学习完的知识**和**展示我知识的学习程度**,我写下此篇总结,智能体开发需要langchain和langgraph框架,我将从**12个方面**来讲述我所学的langchain的知识

1.langchain入门

2.Model I/O大模型接口

3.Ollama本地大模型部署,

4.提示词PromptTemplate和模型调用方法,

5.Parser输出解析器,

6.LCEL链式调用

7.记忆缓存

8.Tools (Function Calling)工具调用

9.向量化和向量数据库

10.检索增强生成RAG

11.MCP(模型上下文协议Model Context Protocol)

12.Agent智能体

# 1.langchain入门

我们在学习一个新知识之前最好的办法就是去看官方文档,和子github上面的讨论

## langchain概述

langchain是一个胶水框架,将大模型和外界联系起来的一套框架.

主要是六部分:1.Models(模型)

2.Memory(记忆)

3.Retrieval(检索)

4.Chains(链)

5.Agents(智能体)

6.Callback(回调)

## langchain入门使用

```python
model= init_chat_model(
    model="deepseek-v4-pro",
    model_provider="openai",
    api_key="sk-9f5095134676449cbeaa2982d2d77984",
    base_url="https://api.deepseek.com"     )
print(model.invoke("你是谁").content)
```

此为调用大模型的核心代码,在api_key中可以把key**存储在env中**

```python
load_dotenv(encoding='utf-8',override=True)
```

使用此段代码读env文件之后将**api_key=os.getenv('DEEPSEEK_API_KEY')**修改

在大模型中可以使用**ChatOpenAI**和**init_chat_model**来调用大模型区别如下               ↓↓↓

![image-20260526143620068](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260526143620068.png)

# 2.Model I/O (大模型接口)

## 概述

是与大模型**进行交互**的核心组件

Model I/O：**标准化**各个大模型的**输入**和**输出**，包含输入模版，模型本身和格式化输出。

Model I/O三件套:输入,处理,输出.

**输入提示(Format),调用模型(Predict),输出解析(Parse)**

langchain中将大模型分为以下几种↓↓↓↓↓↓

![image-20260526144739533](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260526144739533.png)

使用init_chat_model会有几种参数,使用其他的也有一些关键的如model,api_key,temperature之类的(除ollama本地部署的)

![image-20260526144855071](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260526144855071.png)

大模型返回的是一个**AIMessage对象**

# 3.Ollama本地大模型部署

ollama是一个部署大模型的工具

Ollama Hub玩**模型**

核心调用代码

```python
model= ChatOllama(base_url="http://127.0.0.1:11434", model="qwen:4b",reasoning= False)
```

注意调用的模型要已经在**本地安装**的

# 4.提示词PromptTemplate和模型调用方法

## 提示词Prompt

四大角色:将消息分为不同角色（如用户、助手、系统等），设置功能边界，增强交互的复杂性和上下文感知能力

系统提示词**SystemMessage**:为AI设定角色↓↓↓↓↓↓![image-20260526150038176](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260526150038176.png)

人类消息HumanMessage指人类提出来的问题

模型消息AIMessage指AI的回复

工具消息ToolMessage(v1.0)/FunctionMessage(v0.3)指调用的工具消息

## 模型调用方法

共有六种模型调用方法

1.普通同步调用invoke(源代码:lesson2/LLM_Invoke)

```python
response= model.invoke(messages)
```

普通调用，处理单条输入，等待LLM完全推理完成后再返回调用结果

2.普通异步调用ainvoke(源代码:lesson2/LLM_ainvoke)

```python
async def async_invoke_call():
    response= await model.ainvoke(messages)
    print(type( response))
    print(response.content)
    print(response.content_blocks)
```

防止阻塞,性能能快一点注意异步使用async和await

3.流式同步调用stream(源代码:lesson2/LLM_stream)

```python
response = model.stream(messages)
for chunk in response:
    # 刷新缓冲区 (无换行符，缓冲区未刷新，内容可能不会立即显示)
    print(chunk.content, end="",flush=True)
```

像流水一样输出,一块一块的

4.流式异步调用astream

```python
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
```

**注意**:在遍历的时候**使用async for不需要使用await****因为astream输出的response是装饰器类型

5.同步批处理batch

```python
response= model.batch(question)
print(type(response))
for q,r in zip(question,response):
    print(f"问题{q}\n回答{r.content}\n===========================")
```

同时处理多个问题

6.异步批处理abatch

```python
async def async_batch_call():
    response= await model.abatch(questions)
    print(type(response))
    for q,r in zip(questions,response):
        print(f"问题{q}\n回答{r.content}\n===========================")
```

![image-20260526151331186](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260526151331186.png)

## PromptTemplate提示词模板

使用提示词模板来**传递输入**,更清晰地**表达用户意图**，更好地**利用模型能力**

### 提示词模板分类

**1.PromptTemplate：文本生成模型提示词模板，用字符串拼接变量生成提示词**

**2.ChatPromptTemplate：聊天模型提示词模板，适用于如 gpt-3.5-turbo、gpt-4 等聊天模型**

3.FewShotPromptTemplate：(了解):少样本学习提示词模板， 构建一个Prompt其中包含多个示例，可以自动将这些示例格式化并插入到主Prompt 中形成样本提示模板，通过在给模型的最终输入中筛入一些示例，来教模型如何回答

4.PipelinePrompt(了解)：管道提示词模板，用于把几个提示词组合在一起使用

### 常用模板和核心方法

0.3版本和1.0版本不同

0.3

```python
messages = [
    SystemMessage(content="你叫小问，是一个乐于助人的AI人工助手"),
    HumanMessage(content="你是谁")
]
```

1.0

```python
message1= [
    ("system","你叫小问，是一个乐于助人的AI人工助手")
    ("user","你是谁")
]
```

#### PromptTemplate文本提示词模板

出自lesson2/PromptTemplate

```python
template = PromptTemplate(
    template="你是一个专业的{role}工程师，请回答我的问题给出回答，我的问题是：{question}",
    input_variables=['role', 'question']
)
prompt = template.format(role="Java", question="写一段简单的循环")
model1= model.invoke(prompt)
```

使用构造方法👆👆👆,format是初始化使用

input_variables和partial_variables的使用如下👇👇👇出自lesson3/PromptTemplate_conminandgo_on

```python
tem= PromptTemplate(template="你是{role}",
                    input_variables=['role'],
                    partial_variables={'role':'语文老师'}
                    )
print(tem.format())
result= tem.format(role="数学老师")
```

**partial_variables属性相当于提前预设好值**

**input_variables属性相当于占位符**

使用from_template方法

```python
tem1= PromptTemplate.from_template("你是{role}")
result1= tem1.format(role="语文老师")
```

除了format方法外还有invoke()和partial()方法也可格式化提示词模板

**区别:**format是转化成**字符串**,invoke是转化成**PromptValue 对象**，可以用 .to_string() 或 .to_messages() 查看内容,而partial是格式化提示词模板为一个**新的提示词模板**，可以继续进行格式化,在lesson3种有体现

##### 组合提示词

```python
tem2= PromptTemplate.from_template("你是{gender}")
tem3= PromptTemplate.from_template("你是{role}")
all_tem= tem2+tem3
all= all_tem.format(gender="男",role="语文老师")
print(all)
```

#### ChatPromptTemplate对话提示词模板

在代码lesson3/ChatPromptTemplate中体现

```python
tem1=  ChatPromptTemplate (
    [
        ("system","你是一个{role}"),
        ("human","{question}")
    ]
)
#result = tem1.format(role="Java", question="写一段简单的循环")
result= tem1.format(**{"role":"Java", "question":"写一段简单的循环"})
result1 = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("human", "{question}")
])
```

使用构造方法和from_messages都在此

**三种定义ChatPromptTemplate**的格式**元组,字典,Message对象**

### MessagesPlaceholder消息占位符提示词模板

显性使用和隐形使用

```python
prompt = ChatPromptTemplate.from_messages([
    # 添加一条系统消息，设定 AI 的角色或行为准则
    ("system", "你是一个资深的Python应用开发工程师，请认真回答我提出的Python相关的问题"),
    # 插入 memory 占位符，用于填充历史对话记录（如多轮对话上下文）
    MessagesPlaceholder("memory"),
    # 添加一条用户问题消息，用变量 {question} 表示
    ("human", "{question}")
])
# 调用 prompt.invoke 来格式化整个 Prompt 模板
# 传入的参数中：
# - memory：是一组历史消息，表示之前的对话内容（多轮上下文）
# - question：是当前用户的问题
prompt_value = prompt.invoke({
    "memory": [
        # 用户第一轮说的话
        HumanMessage("我的名字叫亮仔，是一名程序员111"),
        # AI 第一轮的回应
        AIMessage("好的，亮仔你好222")
    ],
    # 当前问题：结合上下文，测试模型是否记住了用户名字
    "question": "请问我的名字叫什么？"
})
```

隐形就是

```python
prompt = ChatPromptTemplate.from_messages([
    # 占位符，用于插入对话“记忆”内容，即之前的聊天记录（历史上下文）
    ("placeholder", "{memory}"),
    # 系统消息，用于设定 AI 的角色 —— 是一个资深的 Python 应用开发工程师
    ("system", "你是一个资深的Python应用开发工程师，请认真回答我提出的Python相关的问题"),
    # 用户当前提问，使用变量 {question} 进行动态填充
    ("human", "{question}")
])
```

## 外部提示词

从外部导入提示词如json文件或者yaml文件

json文件

```python
from langchain_core.prompts import load_prompt

template = load_prompt("prompt.json", encoding="utf-8")
print(template.format(name="张三", what="搞笑的"))
# 请张三讲一个搞笑的的故事
```

yaml文件

```python
from langchain_core.prompts import load_prompt

template = load_prompt("prompt.yaml", encoding="utf-8")
print(template.format(name="年轻人", what="滑稽"))
# 请年轻人讲一个滑稽的故事
```

tips:导入import warnings可以忽略部分警告

如:

```python
warnings.filterwarnings("ignore",
                        message="Core Pydantic V1 functionality isn't compatible with Python 3.14")
```

忽略版本过高警告

# 5.Parser输出解析器

是一种可以**指定格式输出**的**工具**

将大语言模型的原始输出内容解析为如JSON、XML、YAML等结构化数据

方法一**:parse**

```python
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
...
tem= ChatPromptTemplate.from_messages([
    ("system", "你是{role},请以JSON格式回答我"),
    ("human", "{question}")
])
...
parse= JsonOutputParser()
a= parse.invoke(model1)
```

重点在于要使用json格式回复

输出也是json格式

方法二:**主要规定输出方式**

意思就是**规定一个类**用此类来规定**输出格式**

```python
class Person(TypeDict):
    time: str = Field(description="时间")#
    person: str = Field(description="人物")
    event: str = Field(description="事件")
```

规定输出的类,可以修改一下来让它更规范,添加**强类型字典**

```python
class Person(TypeDict):
    time: Annotated[str,"时间"] #
    person: Annotated[str,"人物"]
    event: Annotated[str,"事件"]
```

```python
parse1= JsonOutputParser(pydantic_object= Person)
format_instruction= parse1.get_format_instructions()
chat_promat= ChatPromptTemplate([
    ("system", "你是一个新闻编辑，请以JSON格式回答我"),
    ("human", "生成一个{topic}的新闻,{format_instruction}")
])
promat= chat_promat.format(topic="中国与美国", format_instruction=format_instruction)
model2= model.invoke(promat)
```

# 6.LCEL链式调用

## Runnable

Runnable相当于统一的接口

**Runnable 是 LangChain 核心抽象接口(定义在 langchain_core.runnables)统一组件调用方式，支持 LCEL 组合，适配同步 / 异步、流式、批量等场景，是构建工作流的基础**

## LECLn

管道符,将多个Runnable对象组合起来

核心思想：使用管道操作符 |  将多个Runnable对象，像拼积木一样组合起来。

chain = prompt | model | output_parser

## 链式调用常用的5种基础用法

### RunnableSequence-顺序链

最基础

```python
chain= chat_prompt|model|parser
result= chain.invoke({"role":"语文老师","question":"静夜思的作者是谁"})
logger.info(result)
```

按照顺序连住

### RunnableBranch-分支链

同时提问多个问题分别解决时使用

```python
....
chain= RunnableBranch(
    (lambda x : language_dertermine(x)=="english",
    english_prompt | model | parser),
    (lambda x : language_dertermine(x)=="korean",
    korean_prompt | model | parser),
    (lambda x : language_dertermine(x)=="japanese",
    japanese_prompt | model | parser),
    default_handler
)
....
```

### RunnableSerializable-串行链

先执行一个chain在执行另一个chain

```python
full_chain= chain1|(lambda content: {"input": content}) |chain2
```

### RunnableParallel-并行链

同时执行两个chain合并输出一个结果

```python
parallel_chain = RunnableParallel({
    "chinese": chain1,
    "english": chain2
})
result = parallel_chain.invoke({"topic": "langchain"})
```

### RunnableLambda-函数链

在两个chain加个函数(函数要套一下RunnableLambda,虽然不会报错,但是为了规范要使用)

```python
debug_node= RunnableLambda(debug_print)
...
full_chain = chain1 | debug_node | chain2

result1 = full_chain.invoke({"topic": "langchain"})
logger.info(f"最终结果111:{result1}")
```

# 7.记忆缓存

**记忆缓存是聊天系统中的一个重要组件，用于存储和管理对话的上下文信息。它的主要作用是让AI助手能够”记住”之前的对话内容，从而提供连贯和个性化的回复。**

常用消息记忆组件

![image-20260518155501532](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260518155501532.png)

更推荐Redis和Elasticsearch

## 用redisStack作为存储

**RedisStack = 原生Redis + 搜索 + 图 + 时间序列 + JSON + 概率结构 + 可视化工具 + 开发框架支持**

创建会话历史的方法

```python
def get_session_history(session_id: str) -> RedisChatMessageHistory:
    """获取或创建会话历史（使用 Redis）"""
创建 Redis 历史对象
    history = RedisChatMessageHistory(
        session_id=session_id,
        url=REDIS_URL,
        # ttl=3600  # 注释：关闭自动过期，避免重启后数据被清理
    )
return history
```

创建带历史的链

```python
#创建带历史的链

chain = RunnableWithMessageHistory(
    prompt | llm,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)
```

```python
# 配置
# session_id 就是登录大模型的各自帐户，类似登录手机号码，各不相同
config = RunnableConfig(configurable={"session_id": "user-001"})
```

```python
# 主循环
print("开始对话（输入 'quit' 退出）")
while True:
    question = input("\n输入问题：")
    if question.lower() in ['quit', 'exit', 'q']:
        break

    response = chain.invoke({"question": question}, config)
    logger.info(f"AI回答:{response.content}")

    # 等同于redis-cli的SAVE命令，强制写入dump.rdb
    redis_client.save()
```

# 8.Tools (Function Calling)工具调用

通过 Tool（工具）机制，可以让模型具备“调用外部函数”的能力，使其能够与外部系统、API 或自定义函数交互，从而完成仅靠文本生成无法实现的任务

1.访问实时数据

2.执行某种工具类/辅助类操作

泳道图

![image-20260518220333933](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260518220333933.png)

使用案例看langchain中的工具调用

# 9.向量化和向量数据库

![image-20260519084912111](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260519084912111.png)

![image-20260519084940028](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260519084940028.png)

## 向量数据库

一种专门用于**存储、管理和检索向量数据**（即高维数值数组）的数据库系统。

其核心功能是通过高效的索引结构和**相似性**计算算法，支持大规模向量数据的快速查询与分析，向量数据库维度越高，查询精准度也越高，查询效果也越好。

<u>向量数据做的检索不同于其他数据库,是做一个相似性的算法,而不是精确匹配</u>

## Redis存储向量数据

```python
....
doc= [Document(page_content=text, metadata={"source": "manual"}) for text in texts]
vector_store = Redis.from_documents(
    documents=doc,
    embedding=embeddings,
    redis_url="redis://localhost:26379",  # 替换为你的 Redis 地址
    index_name="my_index11",               # 向量索引名称
)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})
results = retriever.invoke("LangChain 和 Redis 怎么结合？")
for res in results:
    print(res.page_content)
```

**为什么要使用Document对象?**

![image-20260519102309613](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260519102309613.png)

doc是一个装满Document对象的列表

把数据存储进redis数据库的核心代码👇👇👇

```python
vector_store = Redis.from_documents(
    documents=doc,
    embedding=embeddings,
    redis_url="redis://localhost:26379",  # 替换为你的 Redis 地址
    index_name="my_index11",               # 向量索引名称
)
```

# 10.检索增强生成RAG

## RAG

RAG技术就像给AI大模型装上了**「实时百科大脑」**，为了让大模型获取足够的上下文，以便获得更加广泛的信息源，通过先查资料后回答的机制，让AI摆脱传统模型的”知识遗忘和幻觉回复”困境

通过**引入外部知识**源来增强LLM的输出能力，传统的LLM通常基于其训练数据生成响应，但这些数据可能过时或不够全面。**RAG允许模型在生成答案之前，从特定的知识库中检索相关信息，从而提供更准确和上下文相关的回答**

### RAG使用方法

#### 索引

![image-20260519155736761](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260519155736761.png)

![image-20260519155742106](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260519155742106.png)

#### 检索

![image-20260519155918833](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260519155918833.png)

## RAG标准流程(重点)

**在RAG准备阶段，LangChain通过文档加载器对各种格式的文档进行加载，转换为LangChain中的文档对象**

**对文档对象进行分割，根据分割规则，分割成文档片段**

**将文档片段通过文本嵌入模型组件，转换为向量，通过向量数据库组件，保存到向量数据库**

**在RAG的使用阶段，用户首先提出问题，使用文本嵌入模型组件，将提问文本转换为向量数据，通过向量数据库检索器组件，进行相似性检索，返回关联的文本片段**

**将相关的文档片段内容渲染到提示词模板中，作为提问问题的上下文传递给大模型，在上下文里做“阅读-理解-整合-生成”，最后把整理好的答案返回给用户**

**RAG的核心卖点正是让生成模型利用检索到的外部知识再做一次深加工，从而给出连贯、准确且带引用的回答**

## 文档加载器

```python
from langchain_community.document_loaders import TextLoader
file_url=r"D:\develop\LangChain\检索增强生成RAG\assets\sample.txt"
encoding = "utf-8"

docs= TextLoader(file_url, encoding).load()
print(docs)
```

## 文本分割器

为了节省token和避免无法接受大量数据的问题使用文本分割器

### 常用分割器

RecursiveCharacterTextSplitter递归按字符分割文本

TokenTextSplitter按照token数量分割

### **函数**

split_text()：将文本字符串分割成字符串列表
split_documents()：将Document对象列表分割成更小文本片段的Document对象列表
create_documents()：通过字符串列表创建Document对象

### 核心参数

#### 核心参数

![image-20260519165711379](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260519165711379.png)

案例代码查看langchain/检索增强生成RAG/RecursiveTestSplitter

### 小项目(AI智能运维)

代码查看langchain/检索增强生成RAG/EmbeddingRAG

# 11.MCP(模型上下文协议Model Context Protocol)

**大模型版的OpenFeign，OpenFeign用于微服务之间通讯，MCP用于大模型之间通讯**

**提供了一种标准化的方式来连接 LLMs 需要的上下文，MCP 就类似于一个 Agent 时代的 Type-C协议，希望能将不同来源的数据、工具、服务统一起来供大模型调用**

## MCP架构知识

**<u>MCP遵循客户端-服务器架构</u>**

MCP 主机（MCP Hosts）：发起请求的 AI 应用程序，比如聊天机器人、AI 驱动的 IDE 等。

MCP 客户端（MCP Clients）：在主机程序内部，与 MCP 服务器保持 1:1 的连接。

MCP 服务器（MCP Servers）：为 MCP 客户端提供上下文、工具和提示信息。

本地资源（Local Resources）：本地计算机中可供 MCP 服务器安全访问的资源，如文件、数据库。

远程资源（Remote Resources）：MCP 服务器可以连接到的远程资源，如通过 API 提供的数据

### 两种模式

STDIO(标准输入/输出):支持标准输入和输出流进行通信，主要用于本地集成、命令行工具等场景(大部分对内)

SSE (Server-Sent Events):流式输出,支持使用 HTTP POST 请求进行服务器到客户端流式处理，以实现客户端到服务器的通信(大部分对外)

![image-20260522081641006](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260522081641006.png)

案例代码请看langchain/MCP

# 12.Agent智能体

**Agent：决策者 = 如何使用这些能力**

Agent 是一个决策引擎:

1.决定什么时候调用哪个 Tool

2.根据上下文决定下一步做什么

3.处理 Tool 返回的结果并决定是否需要继续调用其他 Tool

**Agent 的核心是 推理 + 行动（Reason + Act），也就是 ReAct 模式**

![image-20260522113054169](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260522113054169.png)

大部分智能体主要是由**大模型,工具.提示词**组合成智能体
