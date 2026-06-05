# Langchain进阶2

## PromptTemplate文本提示词模板

partial_variables属性相当于提前预设好值

input_variables属性相当于占位符

```python
from langchain_core.prompts import PromptTemplate
tem= PromptTemplate(template="你是{role}",
                    input_variables=['role'],
                    partial_variables={'role':'语文老师'}
                    )
print(tem.format())
result= tem.format(role="数学老师")
print(result)
```

#### 使用from_template方法

```python
##from_template方法`
`tem1= PromptTemplate.from_template("你是{role}")`
`result1= tem1.format(role="语文老师")`
`print(result1)`


```

#### 组合提示词

```python
tem2= PromptTemplate.from_template("你是{gender}")
tem3= PromptTemplate.from_template("你是{role}")
all_tem= tem2+tem3
all= all_tem.format(gender="男",role="语文老师")
print(all)
```

#### 三种提示词主要方法

1.format,在上面

2.**invoke,返回的是一个 PromptValue 对象,可以用 .to_string() 或 .to_messages() 查看内容**

```python
from langchain_core.prompts import PromptTemplate
tem= PromptTemplate.from_template("你是{role},请回答{question}")
tem1= tem.invoke({"role":"语文老师", "question":"你叫什么"})
print(type(tem1))#<class 'langchain_core.prompt_values.StringPromptValue'>
print(tem1.to_string())#你是语文老师,请回答你叫什么
print(tem1.to_messages())#[HumanMessage(content='你是语文老师,请回答你叫什么', additional_kwargs={}, response_metadata={})]
print(type(tem1.to_messages()))#<class 'list'>
```

***注意:在invoke里面使用({"====":"==="}),具备format的功能,同时也可使用提取HumanMessage之类的信息,方式变多***

3.partial：格式化提示词模板为一个新的提示词模板，可以继续进行格式化

```python
tem2= tem.partial(role="数学老师")
print(type(tem2))#<class 'langchain_core.prompts.prompt.PromptTemplate'>
result= tem2.format(question="你叫什么")
print(result)#你是数学老师,请回答你叫什么
print(type(result))#<class 'str'>
```

**partial就是 “先固定模板里一部分变量，返回一个新的‘半成品模板’，后面只用传剩下的变量,后续记得使用format来填充完整才可使用**

**使用场景属于提前固定一个数值,后续填充完整数据**

## ChatPromptTemplate对话提示词模板

ChatPromptTemplate更适合使用结构化聊天,简单来说,对于对话功能友好,为与现代聊天模型的交互提供了一种上下文丰富和会话友好的方式

是一个元组类型(tuple)

**使用构造方法构造**

```python
tem1=  ChatPromptTemplate (
    [
        ("system","你是一个{role}"),
        ("human","{question}")
    ]
)#构造实例对象
#result = tem1.format(role="Java", question="写一段简单的循环")
result= tem1.format(**{"role":"Java", "question":"写一段简单的循环"})
```

主要有两种方式第一种更普通

第二种(主要利用解包)更好一点

**调用from_messages(常用)*****

```python
result1 = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("human", "{question}")
])
prompt_value2 = result1.invoke({"role": "python开发工程师", "question": "堆排序怎么写"})
```

#### ChatPromptTemplate 三大类型

1.元组

```python
chatPromptTemplate = ChatPromptTemplate(
    [
        ("system", "你是一个AI开发工程师，你的名字是{name}。"),
        ("human", "你能帮我做什么?"),
        ("ai", "我能开发很多{thing}。"),
        ("human", "{user_input}"),
    ]
)
```

2.字典

```python
chat_prompt = ChatPromptTemplate.from_messages(
    [
        {"role": "system", "content": "你是AI助手，你的名字叫{name}。"},
        {"role": "user", "content": "请问：{question}"}
    ]
)
```

3.Message对象

```python
chat_prompt = ChatPromptTemplate(
    [
        SystemMessage(content="你是AI助手，你的名字叫{name}。"),
        HumanMessage(content="请问：{question}")
    ]
)
```

## MessagesPlaceholder消息占位符提示词模板

是一个连接记忆的传递上下文的工具

有两种写法

第一种显式使用MessagesPlaceholder

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

MessagesPlaceholder("memory")

第二种隐式使用MessagesPlaceholder

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

# Parser输出解析器

是一种可以指定格式输出的工具

将大语言模型的原始输出内容解析为如JSON、XML、YAML等结构化数据

## 输出解析器两大方法

方法一:parse

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

方法二:主要规定输出方式

```python
from pydantic import BaseModel,Field
class Person(BaseModel):
    """
    定义一个新闻结构化的数据模型类
    属性:
        time (str): 新闻发生的时间
        person (str): 新闻涉及的人物
        event (str): 发生的具体事件
    """
    time: str = Field(description="时间") #
    person: str = Field(description="人物")
    event: str = Field(description="事件")
parse1= JsonOutputParser(pydantic_object= Person)
format_instruction= parse1.get_format_instructions()
chat_promat= ChatPromptTemplate([
    ("system", "你是一个新闻编辑，请以JSON格式回答我"),
    ("human", "生成一个{topic}的新闻,{format_instruction}")
])
promat= chat_promat.format(topic="中国与美国", format_instruction=format_instruction)
model2= model.invoke(promat)
print(model2.content)
print("==========================")
response1= parse1.invoke(model2)
print(response1)
```

输出格式具有 time,person,event的属性

## 进阶

#### TypedDict

TypedDict是强类型字典

Annotated是强类型注解工具

以此来规定类型如:

```python
Age = Annotated[int, "年龄，范围0-150"]

class Person(TypedDict):
    name: str
    age: int
    age2:Age

p = Person(name="z3",age=111,age2=188)
print(p)
```

只有一个规定作用,和python规定类型一样,没有强制作用

```python
class Animal(TypedDict):
    name:Annotated[str,"动物名称"]
    emoji:Annotated[str,"动物emoji表情"]
class AnimalList(TypedDict):
    animals:Annotated[list[Animal],"动物和表情列表"]

messages= [{"role":"system","content":"生成三种动物,以及他们的emoji表情"}]

llm_with_structured_output= llm.with_structured_output(AnimalList)
result= llm_with_structured_output.invoke(messages)
print(result)
```

#### Pydantic

```python
import os
from langchain_classic.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel,Field,field_validator
from langchain_core.output_parsers import PydanticOutputParser
from loguru import logger


load_dotenv(encoding='utf-8',override=True)

model = init_chat_model(
    model="deepseek-v4-pro",
    model_provider="openai",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=2.0
)
class Product(BaseModel):
    name: str = Field(description="产品名称")
    price: int = Field(description="产品价格")
    description: str = Field(description="产品描述")
    @field_validator("description")
    def validate_description(cls, value):
        if len(value) < 10:
            raise ValueError("描述长度不能小于10")
        return value
parser=PydanticOutputParser(pydantic_object=Product)
format_instruction= parser.get_format_instructions()
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个AI助手，你只能输出结构化的json数据\n{format_instructions}"),
    ("human", "请你输出标题为：{topic}的新闻内容")
])
prompt= prompt_template.format(topic="中国新闻",format_instructions=format_instruction)

result= model.invoke(prompt)
response = parser.invoke(result)
```

属于进阶知识
