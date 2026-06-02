# Agent智能体

agent是决策大脑,tools是工具

**Agent：决策者 = 如何使用这些能力**

Agent 是一个决策引擎:

1.决定什么时候调用哪个 Tool

2.根据上下文决定下一步做什么

3.处理 Tool 返回的结果并决定是否需要继续调用其他 Tool

**Agent 的核心是 推理 + 行动（Reason + Act），也就是 ReAct 模式**

![image-20260522113054169](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260522113054169.png)

大部分智能体主要是由大模型,工具.提示词组合成智能体

## ReAct 模式

```python
agent = create_agent(
    llm,
    tools=[search_products, check_inventory],
    system_prompt="""你是电商助手，遵循ReAct模式：
    1. 先推理用户需求
    2. 选择合适的工具执行操作
    3. 基于工具结果进行下一步推理
    4. 重复直到获得完整答案

    保持推理步骤简洁明了。"""
)
```

制造智能体的操作 create_agent(model,tools,system_prompt)

注意要加遵循ReAct模式

```python
result1 = agent.invoke({
    "messages": [{"role": "user", "content": "查找当前最受欢迎的无线耳机并检查是否有库存"}]
})

print("\n" + "=" * 40)
print("📊 最终结果:")
for msg in result1['messages']:
    if hasattr(msg, 'content'):
        print(f"{msg.__class__.__name__}: {msg.content}")
```

```python
# # 详细追踪ReAct循环过程
def track_react_cycle(messages):
    print("ReAct循环步骤分析:")
    step = 1
    for i, msg in enumerate(messages):
        msg_type = msg.__class__.__name__
        if msg_type == "AIMessage" and hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"\n🔄 步骤{step}: Reasoning + Acting")
            for tool_call in msg.tool_calls:
                print(f"   🛠️  工具调用: {tool_call['name']}({tool_call['args']})")
            step += 1
        elif msg_type == "ToolMessage":
            print(f"   📋  观察结果: {msg.content[:80]}...")
        elif msg_type == "AIMessage" and not (hasattr(msg, 'tool_calls') and msg.tool_calls):
            print(f"\n✅ 最终回答: {msg.content}")

# 追踪案例1的ReAct循环
track_react_cycle(result1['messages'])
```



