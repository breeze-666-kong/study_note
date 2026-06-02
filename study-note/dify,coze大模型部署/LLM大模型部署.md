# 大模型部署

## 大模型部署好处

数据安全,可控制版本,成本预算可控,网络延迟低,可观测故障

![image-20260514105248817](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260514105248817.png)

使用**Xinference托管平台,大语言模型,嵌入模型,重排序模型都可用

Docker主要是使用cpu,而xinference主要是使用GPU所以xinference不部署在Docker上(或者安装额外工具使其可以使用GPU资源)

## Dify私有化部署

### docker的安装

首先安装Xshell(**相当于遥控器来远程使用指令操控linux系统**)

建立CVM云服务器,得到用户名密码,端口,连接腾讯云或阿里云

在此安装docker(**只需使用打包的镜像便可直接使用软件的部署工具**)

> #更新软件包
> sudo apt update
>
> sudo apt upgrade
>
> #安装docker依赖
> sudo apt install software-properties-common
>
> sudo apt-get install ca-certificates curl gnupg lsb-release
>
> #添加Docker官方GPG密钥
> curl -fsSL http://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo apt-key add -
>
> #添加Docker软件源（输入后根据提示按Enter）
> sudo add-apt-repository "deb [arch=amd64] http://mirrors.aliyun.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable"
>
> #安装docker（输入后根据提示输入 y ）
> sudo apt-get install docker-ce docker-ce-cli containerd.io --fix-missing

![image-20260514114154843](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260514114154843.png)

成功!!!

### Dify安装

本地部署两种方案 1离线压缩包下载和2在gitee上使用

使用方法1 需要使用xftp(**是文件传输工具，可以把 Windows 本地文件上传到 Linux 服务器，也能从服务器下载文件到 Windows**) 

首先上传可能出错,权限不够

> 方式1：赋予指定用户指定目录的完全权限（使用777）
>
> 在Ubuntu终端xshell执行：sudo chmod -R 777 /目标目录的完整路径
>
> sudo chmod -R 777 /opt/dify

> #进入dify目录,在opt目录下执行：
> cd ./dify
> #解压
> sudo tar -zxvf dify-0.15.5.tar.gz
>
> cd /opt/dify/dify-0.15.5
>
> pwd # 输出 /opt/dify/dify-0.15.5

首先进入docker

cd /opt/dify/dify-0.15.5/docker

复制环境配置文件

sudo cp .env.example .env

docker启动

sudo docker compose up -d

![image-20260514121535812](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260514121535812.png)

出现错误(网络无法访问)重新配置

进行镜像配置

sudo vi /etc/docker/daemon.json

进入后在空白处先摁**i**再输入

{
    "registry-mirrors": [
    "https://docker.unsee.tech",
    "https://dockerpull.org",
    "https://docker.1panel.live",
    "https://dockerhub.icu",
    "https://docker.m.daocloud.io",
    "https://docker.nju.edu.cn",
    "https://registry.docker-cn.com",
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com",
    "https://mirror.baidubce.com",
    "https://5tqw56kt.mirror.aliyuncs.com",
    "https://docker.hpcloud.cloud",
    "http://mirrors.ustc.edu.cn",
    "https://docker.chenby.cn",
    "https://docker.ckyl.me",
    "http://mirror.azure.cn",
    "https://hub.rat.dev"]
}

后摁esc后输入":wq"

检查容器是否正常运行

sudo docker compose ps

之后打开网页

http://your_server_ip/install (your_server_ip是指![image-20260514144032303](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260514144032303.png))

ps:前提先配置安全组

### 部署大语言模型

首先需要登录AutoDL来租算力再算力市场组一个GPU来使用xinference

大概流程在自定义服务中找到用的数据在xshell上连接

![image-20260514170555536](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260514170555536.png)

之后在xshell上面部署从创建初始化激活验证conda在conda上部署XInference之后启动在自定义地址进入

在xshell上下载大语言模型(时间巨巨巨巨巨久)

### 部署Embedding模型

直接在同一xshell上部署

### 部署Rerank模型

直接在同一xshell上部署

### Dify对接XInference

直接在同一xshell上部署