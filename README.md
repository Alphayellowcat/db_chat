# 数据库聊天机器人

这是一个基于 LangChain 的智能数据库助手，支持 OpenAI 和 Deepseek API，允许用户使用自然语言与 MySQL 数据库交互。

## 功能特点

- 多 API 支持：可切换使用 OpenAI 或 Deepseek API
- 智能路由系统：自动识别查询类型并选择最佳处理方式
- 探索性查询：帮助用户发现和理解数据结构
- 数据分析：支持复杂的数据统计和分析
- 可视化支持：自动生成数据可视化配置
- 流式响应：实时展示生成结果
- Web 界面：基于 Streamlit 的友好交互界面
- 错误恢复：自动重试和错误处理机制

## 项目结构

```
db_chat/
├── app.py            # Streamlit Web 界面
├── chat_with_db.py   # 核心聊天逻辑
├── config.py         # 配置管理
├── test_openai.py    # API 测试工具
├── .env              # 环境变量配置
├── .env.example      # 环境变量示例
└── requirements.txt  # 项目依赖
```

## 主要功能模块

### 1. 智能路由系统
- sql_query：数据库查询和探索
- report：数据分析报告生成
- visualization：数据可视化
- chat：一般对话交互

### 2. 查询类型
- 数据库结构解释
- 探索性数据查询
- 统计分析查询
- 可视化数据查询

### 3. 交互界面
- 命令行界面：直接终端交互
- Web 界面：Streamlit 应用
- API 切换：支持在不同 API 间切换

## 安装和配置

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 配置环境变量：
```bash
cp .env.example .env
# 编辑 .env 文件，填入：
# - 数据库配置
# - OpenAI API 配置
# - Deepseek API 配置
```

## 使用方式

### 1. Web 界面启动：
```bash
streamlit run db_chat/app.py
```

### 2. 命令行启动：
```bash
python db_chat/chat_with_db.py
```

### 3. API 测试：
```bash
python db_chat/test_openai.py
```

## 使用示例

1. 探索数据：
```
> 帮我看看这些表里有没有销售数据的？
[系统会自动查找相关表和字段，并给出建议]

> 分析一下不同类型专辑的价格有何不同
[系统会生成统计分析并解释结果]
```

2. 数据可视化：
```
> 用柱状图展示各个流派的销售情况
[系统会生成适当的图表配置]
```

## 注意事项

1. API 配置
   - 确保 API 密钥配置正确
   - 检查 API 的可用性和限制

2. 数据库连接
   - 验证数据库连接信息
   - 确保必要的数据库权限

3. 性能考虑
   - 合理使用数据库索引
   - 注意查询的复杂度

## 开发扩展

1. 添加新的 API 支持
2. 自定义提示模板
3. 扩展可视化类型
4. 优化查询性能

## 故障排除

1. API 连接问题
   - 检查网络连接
   - 验证 API 密钥
   - 尝试切换 API

2. 数据库问题
   - 检查连接配置
   - 验证数据库权限
   - 检查 SQL 语法 