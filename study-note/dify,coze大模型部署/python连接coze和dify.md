# Python连接coze和dify

coze---------------连接生成海报工作流,有直接的python代码直接粘贴即可,

```python
handle_workflow_iterator(
    coze.workflows.runs.stream(
        workflow_id=workflow_id,
        parameters={
    "BOT_USER_INPUT": "",
    "img": "{}",
    "prompt": "小女孩在喝汽水",
    "propotion": ""
  },
    )
)
```

核心修改此处,添加提示词

dify-----------------钉钉投诉处理

没有直接的python代码,根据shell命令修改

```python
import requests

api_key = "app-2r5B5Rj6lx4a92GzEMSY1Sqa"
url = "https://api.dify.ai/v1/workflows/run"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "inputs": {"feedback":"为什么不回复"},
    "response_mode": "streaming",
    "user": "abc-123"
}

# 流式接收返回
response = requests.post(url, headers=headers, json=data, stream=True)

for line in response.iter_lines():
    if line:
        print(line.decode("utf-8"))
```

![image-20260514103954723](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260514103954723.png)输入在input中