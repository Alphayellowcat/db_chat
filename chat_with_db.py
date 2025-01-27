# 导入必要的 LangChain 组件
from langchain_core.prompts import ChatPromptTemplate  # 用于创建聊天提示模板
from langchain_core.output_parsers import StrOutputParser  # 用于解析输出
from langchain_core.runnables import RunnablePassthrough  # 用于创建可运行的链
from langchain_openai import ChatOpenAI  # OpenAI的聊天模型接口
from langchain_community.utilities import SQLDatabase  # SQL数据库工具
from config import MYSQL_CONFIG, OPENAI_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
import os
from typing import Iterator
import sys
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import Literal

# 设置API密钥
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def get_schema(db):
    """获取数据库schema信息
    Args:
        db: SQLDatabase实例
    Returns:
        str: 数据库表结构信息
    """
    return db.get_table_info()

def run_query(db, query):
    """执行SQL查询
    Args:
        db: SQLDatabase实例
        query (str): SQL查询语句
    Returns:
        str: 查询结果或错误信息
    """
    try:
        return db.run(query)
    except Exception as e:
        return f"查询执行错误: {str(e)}"

class DBChatBot:
    def __init__(self):
        """初始化数据库聊天机器人
        设置数据库连接、LLM模型和提示模板
        """
        # 创建数据库连接字符串，包含所有必要的连接参数
        db_url = f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}?charset={MYSQL_CONFIG['charset']}"
        self.db = SQLDatabase.from_uri(db_url)
        
        # 初始化 ChatOpenAI 并配置 Deepseek
        self.llm = ChatOpenAI(
            temperature=0,  # 设置为0以获得确定性输出
            model_name=DEEPSEEK_MODEL,
            base_url=DEEPSEEK_BASE_URL,
            streaming=True  # 启用流式响应
        )
        
        # 创建SQL生成提示模板 - 用于将自然语言转换为SQL查询
        self.sql_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个SQL专家。根据用户的问题生成对应的SQL查询。
            规则：
            1. 只返回一个有效的SQL查询语句
            2. 不要返回任何注释或解释
            3. 不要使用markdown格式
            4. 对于数据库概览类的问题，使用 SHOW TABLES 或 查询具体表的内容
            5. 确保生成的是可执行的SQL语句
            
            数据库结构如下：
            {schema}
            """),
            ("human", "生成一个SQL查询来回答这个问题：{question}")
        ])
        
        # 创建响应生成提示模板 - 用于解释SQL查询结果
        self.response_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个数据分析师，需要解释SQL查询的结果。
            规则：
            1. 用通俗易懂的语言解释结果
            2. 如果结果为空，说明可能的原因
            3. 如果有错误，解释错误原因并提供建议
            4. 如果是数据库概览问题，详细解释每个表的用途
            
            数据库结构：
            {schema}
            """),
            ("human", """
            用户问题：{question}
            执行的SQL：{query}
            查询结果：{result}
            
            请解释这个结果：
            """)
        ])
        
        # 修改路由提示模板
        self.router_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个对话意图分析专家。请仔细分析用户的问题属于哪种类型：

            1. sql_query: 需要查询数据库获取信息的问题
               - 查看数据库结构
               - 查询具体数据
               - 了解数据库表的内容
               - 包含"解释"、"查询"、"显示"、"告诉我"等直接查询意图的词语
               
            2. report: 要求对已有信息进行汇总分析的请求
               - 明确要求生成报表
               - 要求对多个查询结果进行总结
               - 包含"总结"、"报表"、"分析报告"等汇总分析意图的词语
               - 需要对历史查询结果进行综合分析
            
            只返回一个JSON对象，格式如下：
            {{"type": "sql_query 或 report"}}
            """),
            ("human", "{question}")
        ])
        
        # 添加报表生成提示模板
        self.report_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的数据分析师，需要根据之前的查询结果生成一份完整的分析报表。
            
            规则：
            1. 报表应该结构清晰，包含标题、概述、详细分析和建议
            2. 使用专业但易懂的语言
            3. 如果有数字变化，要分析趋势
            4. 给出具体的商业建议
            
            数据库结构：
            {schema}
            """),
            ("human", """
            历史查询和结果：
            {history}
            
            请生成一份完整的分析报表：
            """)
        ])
        
        # 添加输出解析器
        self.router_parser = JsonOutputParser()
        
    def process_streaming_response(self, response: Iterator) -> str:
        """处理流式响应
        
        Args:
            response: LLM的流式响应迭代器
            
        Returns:
            str: 完整的响应内容
        """
        full_response = []
        for chunk in response:
            if chunk.content:
                print(chunk.content, end='', flush=True)
                full_response.append(chunk.content)
        print()  # 换行
        return ''.join(full_response)
    
    def process_report_request(self, history: list) -> str:
        """处理报表生成请求"""
        schema = get_schema(self.db)
        history_text = "\n".join([f"问题：{q}\n回答：{a}" for q, a in history])
        
        print("\n生成报表...", flush=True)
        report_response = self.process_streaming_response(
            self.llm.stream(
                self.report_prompt.format(
                    schema=schema,
                    history=history_text
                )
            )
        )
        
        return f"""
=== 数据分析报表 ===
{report_response}
"""

    def chat(self, user_question: str, history: list = None) -> str:
        """处理用户问题并返回答案"""
        if history is None:
            history = []
            
        try:
            # 使用路由器确定问题类型
            router_response = self.process_streaming_response(
                self.llm.stream(
                    self.router_prompt.format(question=user_question)
                )
            )
            question_type = self.router_parser.parse(router_response)["type"]
            
            if question_type == "report":
                return self.process_report_request(history)
            else:
                # 原有的SQL查询处理逻辑
                response = self._process_sql_query(user_question)
                history.append((user_question, response))
                return response
                
        except Exception as e:
            return f"抱歉，处理您的问题时出现错误: {e}"
            
    def _process_sql_query(self, user_question: str) -> str:
        """处理SQL查询请求"""
        schema = get_schema(self.db)
        
        print("\n生成SQL查询...", flush=True)
        if "解释" in user_question and "数据库" in user_question:
            sql_query = "SHOW TABLES;"
        else:
            sql_response = self.process_streaming_response(
                self.llm.stream(
                    self.sql_prompt.format(
                        schema=schema,
                        question=user_question
                    )
                )
            )
            sql_query = sql_response.strip()
        
        print(f"\nSQL查询: {sql_query}")
        
        # 执行SQL查询
        print("\n执行查询...", flush=True)
        result = run_query(self.db, sql_query)
        print(f"查询结果: {result}")
        
        # 使用LLM解释查询结果（流式输出）
        print("\n生成解释...", flush=True)
        final_response = self.process_streaming_response(
            self.llm.stream(
                self.response_prompt.format(
                    schema=schema,
                    question=user_question,
                    query=sql_query,
                    result=result
                )
            )
        )
        
        return f"""
SQL查询: {sql_query}
查询结果: {result}
解释: {final_response}
"""

def main():
    """主函数：创建聊天机器人实例并运行交互循环"""
    chatbot = DBChatBot()
    print("欢迎使用数据库聊天机器人！输入 'quit' 退出。")
    
    history = []  # 用于存储对话历史
    
    while True:
        user_input = input("\n请输入您的问题: ")
        if user_input.lower() == 'quit':
            break
            
        response = chatbot.chat(user_input, history)
        print(f"\n{response}")

if __name__ == "__main__":
    main() 