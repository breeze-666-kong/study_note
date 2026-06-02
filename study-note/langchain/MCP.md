# MCP

MCP是大模型之间连接的一套接口标准,用于连接各个大模型

https://mcp.so/zh

MCP调用网址

**大模型版的OpenFeign，OpenFeign用于微服务之间通讯，MCP用于大模型之间通讯**

**提供了一种标准化的方式来连接 LLMs 需要的上下文，MCP 就类似于一个 Agent 时代的 Type-C协议，希望能将不同来源的数据、工具、服务统一起来供大模型调用**

## MCP架构知识

MCP遵循客户端-服务器架构

### 核心部分

MCP 主机（MCP Hosts）：发起请求的 AI 应用程序，比如聊天机器人、AI 驱动的 IDE 等。

MCP 客户端（MCP Clients）：在主机程序内部，与 MCP 服务器保持 1:1 的连接。

MCP 服务器（MCP Servers）：为 MCP 客户端提供上下文、工具和提示信息。

本地资源（Local Resources）：本地计算机中可供 MCP 服务器安全访问的资源，如文件、数据库。

远程资源（Remote Resources）：MCP 服务器可以连接到的远程资源，如通过 API 提供的数据

### 两种模式

STDIO(标准输入/输出):支持标准输入和输出流进行通信，主要用于本地集成、命令行工具等场景(大部分对内)

SSE (Server-Sent Events):流式输出,支持使用 HTTP POST 请求进行服务器到客户端流式处理，以实现客户端到服务器的通信(大部分对外)

![image-20260522081641006](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260522081641006.png)

## 案例

分为两个部分(sever服务端,client客户端)

**server服务端**

```python
import httpx
import json
from mcp.server import FastMCP
from loguru import logger
mcp= FastMCP(name="WeatherServerSSE",host="0.0.0.0",port=8080)


@mcp.tool()
def get_weather(city: str) -> str:
    """
    获取天气信息
    """
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": "5e950228baa199738b40bb66c979da3a",
        "units": "metric",
        "lang": "zh_cn"
    }
    response = httpx.get(url, params=params, timeout=30)
    data = response.json()
    logger.info(f"查询{city}天气,{data}")
    return json.dumps(data, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="sse")
```

由fastmcp创建实例对象,制造工具tool获取天气信息方法,

response = httpx.get(url, params=params, timeout=30)接受回复的数据

logger.info(f"查询{city}天气,{data}")将数据转化为json格式返回json字符串

mcp.run(transport="sse")来对应json中的weather

```json
{
  "mcpServers": {
    "weather": {
      "url": "http://127.0.0.1:8080/sse",
      "transport": "sse"
    },
    "fetch": {
      "command": "uvx",
      "args": [
        "mcp-server-fetch"
      ],
      "transport": "stdio"
    }
  }
}
```

**client客户端**

```python
import json

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mcp_adapters.client import MultiServerMCPClient
from loguru import logger
import asyncio
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from openai import api_key, base_url


def load_server(file_path: str="mcp.json"):
    logger.info(f"加载配置文件{file_path}")
    with open(file_path, "r",encoding="utf-8") as f:
        config = json.load(f)
        logger.info(f"配置文件{file_path}加载成功,类型为:{type(config)},{config.get('mcpServers', {})}")
        return config.get("mcpServers", {})
async def run_server():
    """
    该函数会：
    1. 加载 MCP 服务器配置；
    2. 初始化 MCP 客户端并获取工具；
    3. 创建基于 deepseek 的语言模型和代理；
    4. 启动命令行聊天循环；
    5. 在退出时清理资源。
    :return:
    """
    servers = load_server()

    mcp_client= MultiServerMCPClient(servers)
    tools = await mcp_client.get_tools()
    logger.info(f"工具列表:{tools},{len(tools)}个工具,{[t.name for t in tools]}")

    llm = init_chat_model(
        model="qwen-plus",
        model_provider="openai",
        api_key="sk-14fd29f4451449cbb013458844d60fa3",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    logger.info("模型对象成功")
    prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的助手，需要使用提供的工具来完成用户请求。"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
    logger.info("提示词对象成功")
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors="解析用户请求失败，请重新输入清晰的指令"
    )
    logger.info("智能体对象成功")
    logger.info("\n🤖 MCP Agent 已启动，请先输入一个提问给(LLM+MCP)，输入 'quit' 退出")
    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() == "quit":
            break
        try:
            result = await agent_executor.ainvoke({"input": user_input})
            print(f"\nAI: {result['output']}")
        except Exception as exc:
            logger.error(f"\n⚠️  出错: {exc}")


    logger.info("🧹 会话已结束，Bye!")

if __name__ == "__main__":
    # 启动异步事件循环并运行聊天代理
    asyncio.run(run_server())
```

函数

```python
def load_server(file_path: str="mcp.json"):
    logger.info(f"加载配置文件{file_path}")
    with open(file_path, "r",encoding="utf-8") as f:
        config = json.load(f)
        logger.info(f"配置文件{file_path}加载成功,类型为:{type(config)},{config.get('mcpServers', {})}")
        return config.get("mcpServers", {})
```

加载json文件将其变成字典返回

方法run_server运行客户端,asyncio异步是为了更好的性能

```python
mcp_client= MultiServerMCPClient(servers)
tools = await mcp_client.get_tools()
```

第一步是创建mcp对象实例,第二步连接其中的工具

```python
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors="解析用户请求失败，请重新输入清晰的指令"
)
logger.info("智能体对象成功")
logger.info("\n🤖 MCP Agent 已启动，请先输入一个提问给(LLM+MCP)，输入 'quit' 退出")
```

创建智能体