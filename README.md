# 数据库聊天机器人

这是一个基于 LangChain 和 Deepseek 的数据库聊天机器人，允许用户使用自然语言查询 MySQL 数据库。

## 功能特点

- 自然语言转换为 SQL 查询
- 智能解释查询结果
- 支持数据库结构概览
- 错误处理和解释
- 交互式命令行界面

## 项目结构

```
db_chat/
├── config.py          # 配置文件
├── database.py        # 数据库连接管理
├── chat_with_db.py    # 主要聊天逻辑
└── requirements.txt   # 依赖包
```

## 技术栈

- LangChain：用于构建 LLM 应用
- Deepseek：大语言模型
- MySQL：数据库
- Python：编程语言

## 主要组件说明

### 1. ChatPromptTemplate

LangChain 中的提示模板系统，用于构建结构化的提示。项目中使用了两个主要模板：

- SQL生成模板：将用户问题转换为SQL查询
- 响应生成模板：解释SQL查询结果

### 2. SQLDatabase

LangChain 提供的数据库工具，用于：
- 连接数据库
- 执行查询
- 获取数据库结构信息

### 3. ChatOpenAI

LLM 接口，配置为使用 Deepseek 模型，主要用于：
- 生成 SQL 查询
- 解释查询结果

## 使用流程

1. 用户输入自然语言问题
2. 系统获取数据库结构信息
3. LLM 生成对应的 SQL 查询
4. 执行 SQL 查询获取结果
5. LLM 解释查询结果
6. 返回格式化的响应

## 安装和配置

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 配置环境变量：
- 复制 `.env.example` 为 `.env`
- 修改 `.env` 文件中的配置：
```bash
cp .env.example .env
# 编辑 .env 文件，填入实际的配置值
```

## 使用示例

```bash
python chat_with_db.py

> 请输入您的问题: 解释一下数据库
查询: SHOW TABLES;
结果: [表名列表]
解释: [数据库结构解释]

> 请输入您的问题: 查询用户表的前5条记录
查询: SELECT * FROM users LIMIT 5;
结果: [查询结果]
解释: [结果解释]
```

## 注意事项

1. 确保 MySQL 服务器已启动
2. 检查数据库连接配置是否正确
3. 验证 API 密钥是否有效
4. 确保数据库中存在所需的表和数据

## 进阶使用

1. 自定义提示模板
2. 调整 LLM 参数
3. 添加新的查询类型处理
4. 扩展错误处理机制

## 常见问题

1. 数据库连接错误
   - 检查配置信息
   - 确认数据库服务状态

2. API 调用失败
   - 验证 API 密钥
   - 检查网络连接

3. SQL 查询错误
   - 检查数据库权限
   - 验证表结构是否正确 