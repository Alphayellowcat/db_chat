# 导入必要的 LangChain 组件
from langchain_core.prompts import ChatPromptTemplate  # 用于创建聊天提示模板
from langchain_core.output_parsers import StrOutputParser  # 用于解析输出
from langchain_core.runnables import RunnablePassthrough  # 用于创建可运行的链
from langchain_openai import ChatOpenAI  # OpenAI的聊天模型接口
from langchain_community.utilities import SQLDatabase  # SQL数据库工具
from config import MYSQL_CONFIG, OPENAI_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
import os

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
        
    def chat(self, user_question: str) -> str:
        """处理用户问题并返回答案
        
        Args:
            user_question (str): 用户的自然语言问题
            
        Returns:
            str: 包含SQL查询、结果和解释的格式化响应
        """
        try:
            # 获取数据库schema信息
            schema = get_schema(self.db)
            
            # 对于特殊问题的处理（如数据库概览）
            if "解释" in user_question and "数据库" in user_question:
                sql_query = "SHOW TABLES;"
            else:
                # 使用LLM生成SQL查询
                sql_response = self.llm.invoke(
                    self.sql_prompt.format(
                        schema=schema,
                        question=user_question
                    )
                )
                sql_query = sql_response.content.strip()
            
            # 执行SQL查询
            result = run_query(self.db, sql_query)
            
            # 使用LLM解释查询结果
            final_response = self.llm.invoke(
                self.response_prompt.format(
                    schema=schema,
                    question=user_question,
                    query=sql_query,
                    result=result
                )
            )
            
            # 返回格式化的响应
            return f"""
查询: {sql_query}
结果: {result}
解释: {final_response.content}
"""
            
        except Exception as e:
            return f"抱歉，处理您的问题时出现错误: {e}"

def main():
    """主函数：创建聊天机器人实例并运行交互循环"""
    chatbot = DBChatBot()
    print("欢迎使用数据库聊天机器人！输入 'quit' 退出。")
    
    while True:
        user_input = input("\n请输入您的问题: ")
        if user_input.lower() == 'quit':
            break
            
        response = chatbot.chat(user_input)
        print(f"\n{response}")

if __name__ == "__main__":
    main() 