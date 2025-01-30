# 导入必要的 LangChain 组件
import os
import sys
import ast
import json
from typing import Iterator, List, Tuple
import time
import logging

# LangChain相关组件（此处根据你的环境自行替换/调整）
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase

from config import MYSQL_CONFIG, OPENAI_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, OPENAI_BASE_URL, OPENAI_MODEL, DEEPSEEK_API_KEY

# 设置API密钥
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_schema(db) -> str:
    """获取数据库schema信息
    Args:
        db: SQLDatabase实例
    Returns:
        str: 数据库表结构信息
    """
    return db.get_table_info()

def run_query(db, query: str):
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
    
def get_partial_schema(db, max_tables=5, max_columns=10) -> str:
    """
    获取部分数据库结构，避免Prompt过长：
    只展示前 max_tables 张表的前 max_columns 列。
    """
    # 获取所有表
    try:
        tables_str = run_query(db, "SHOW TABLES;")  # 可能返回 "[('table1',), ('table2',)]"
        tables_list = ast.literal_eval(tables_str)  # 安全解析字符串
        all_tables = [t[0] for t in tables_list if t]
    except Exception as e:
        print(f"获取表信息失败: {e}")
        return ""

    partial_info = []
    for i, table_name in enumerate(all_tables):
        if i >= max_tables:
            partial_info.append("\n... 其余表已省略 ...\n")
            break
        # DESCRIBE 表结构
        desc_str = run_query(db, f"DESCRIBE {table_name};")
        try:
            desc_list = ast.literal_eval(desc_str)  # 解析成 Python 对象
        except Exception as e:
            print(f"解析表结构失败: {e}")
            desc_list = []

        # 只保留前 max_columns 列的信息
        limited_desc = desc_list[:max_columns]
        if len(desc_list) > max_columns:
            limited_desc.append(("...", "...", "...", "...", "...", "..."))

        partial_info.append(f"表名: {table_name}\n字段: {limited_desc}\n")

    return "\n".join(partial_info)


class DBChatBot:
    def __init__(self, api_type="openai", timeout: int = 30, max_retries: int = 2):
        """初始化聊天机器人
        
        Args:
            api_type: API类型，"openai" 或 "deepseek"
            timeout: 超时时间（秒）
            max_retries: 最大重试次数
        """
        self.api_type = api_type.lower()
        self.logger = logging.getLogger(__name__)
        
        try:
            # 连接数据库
            self.db = SQLDatabase.from_uri(
                f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@"
                f"{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}"
            )
            
            # 根据 API 类型初始化 LLM
            if self.api_type == "openai":
                self.llm = ChatOpenAI(
                    temperature=0,
                    model_name=OPENAI_MODEL,
                    api_key=OPENAI_API_KEY,
                    base_url=OPENAI_BASE_URL,
                    streaming=True,
                    timeout=timeout,
                    max_retries=max_retries
                )
                self.non_streaming_llm = ChatOpenAI(
                    temperature=0,
                    model_name=OPENAI_MODEL,
                    api_key=OPENAI_API_KEY,
                    base_url=OPENAI_BASE_URL,
                    streaming=False
                )
            else:  # deepseek
                self.llm = ChatOpenAI(
                    temperature=0,
                    model_name=DEEPSEEK_MODEL,
                    api_key=DEEPSEEK_API_KEY,
                    base_url=DEEPSEEK_BASE_URL,
                    streaming=True,
                    timeout=timeout,
                    max_retries=max_retries
                )
                self.non_streaming_llm = ChatOpenAI(
                    temperature=0,
                    model_name=DEEPSEEK_MODEL,
                    api_key=DEEPSEEK_API_KEY,
                    base_url=DEEPSEEK_BASE_URL,
                    streaming=False
                )
            
            # 测试连接
            self._test_llm_connection(self.non_streaming_llm)
            
            # 初始化所有提示模板
            self._initialize_prompts()
            
        except Exception as e:
            self.logger.error(f"初始化失败: {e}")
            raise
    
    def _test_llm_connection(self, llm):
        """测试LLM连接是否正常"""
        try:
            messages = [{"role": "user", "content": "测试连接"}]
            test_response = llm.invoke(messages)
            if not test_response:
                raise Exception("未收到有效响应")
            print("LLM连接测试成功！")
        except Exception as e:
            raise Exception(f"LLM连接测试失败: {e}")
    
    def _ensure_llm_available(self):
        """确保LLM可用，如果不可用则尝试重新初始化"""
        if not self.llm or not self.non_streaming_llm:
            try:
                print("尝试重新连接到LLM服务...", flush=True)
                # 重新初始化非流式LLM
                self.non_streaming_llm = ChatOpenAI(
                    temperature=0,
                    model_name=DEEPSEEK_MODEL,
                    base_url=DEEPSEEK_BASE_URL,
                    streaming=False
                )
                self._test_llm_connection(self.non_streaming_llm)
                
                # 重新初始化流式LLM
                self.llm = ChatOpenAI(
                    temperature=0,
                    model_name=DEEPSEEK_MODEL,
                    base_url=DEEPSEEK_BASE_URL,
                    streaming=True,
                    timeout=30,
                    max_retries=2
                )
            except Exception as e:
                self.logger.error(f"无法连接到LLM服务: {e}")
                raise
    
    def process_streaming_response(self, response: Iterator) -> str:
        """处理流式响应
        
        Args:
            response: LLM的流式响应迭代器
            
        Returns:
            str: 完整的响应内容
        """
        try:
            full_response = []
            for chunk in response:
                if chunk and hasattr(chunk, 'content') and chunk.content:
                    print(chunk.content, end='', flush=True)
                    full_response.append(chunk.content)
            print()  # 换行
            
            if not full_response:  # 如果没有收到任何响应
                raise Exception("未收到有效响应")
                
            return ''.join(full_response)
            
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                return "抱歉，服务响应超时。建议稍后重试或尝试简化您的问题。"
            elif "connection" in error_msg.lower():
                return "抱歉，连接服务失败。请检查网络连接或稍后重试。"
            else:
                return f"处理响应时发生错误: {e}\n建议稍后重试。"
    
    def clean_sql_query(self, sql: str) -> str:
        """清理SQL查询文本，移除可能的Markdown标记等。"""
        return sql.replace('```sql', '').replace('```', '').strip()

    def _clean_json_response(self, response: str) -> str:
        """清理LLM返回的JSON文本，移除可能的markdown标记等"""
        # 移除可能的markdown代码块标记
        cleaned = response.replace('```json', '').replace('```', '').strip()
        # 移除其他可能的格式化字符
        cleaned = cleaned.replace('\n', '').replace('\r', '')
        return cleaned

    def route_and_process(self, inputs: dict) -> str:
        """
        第一层路由 (sql_query / report)，
        如果是 sql_query，则进入第二层路由 (db_structure / sql_data)。
        """
        try:
            self._ensure_llm_available()
        except Exception as e:
            return f"抱歉，LLM服务暂时不可用: {e}\n建议稍后重试或联系管理员。"

        print("正在理解您的问题类型...", flush=True)
        try:
            # 使用已有的非流式LLM实例
            route_response = self.non_streaming_llm.invoke(
                self.router_prompt.format_messages(question=inputs["question"])
            )
            route_text = self._clean_json_response(route_response.content)
            try:
                route_result = json.loads(route_text)
            except json.JSONDecodeError:
                return f"路由返回的不是有效的JSON格式: {route_text}"
        except Exception as e:
            return f"抱歉，服务暂时响应较慢或不可用: {e}\n建议稍后重试。"

        # 确保JSON里至少有"type"
        top_type = route_result.get("type")
        if not top_type:
            return f"无法解析路由结果，缺少 'type' 字段: {route_result}"

        if top_type == "sql_query":
            # 2) 进入第二层路由
            try:
                # 使用已有的非流式LLM实例进行子路由
                sub_route_response = self.non_streaming_llm.invoke(
                    self.sql_router_prompt.format_messages(question=inputs["question"])
                )
                sub_route_text = self._clean_json_response(sub_route_response.content)
                try:
                    sub_route_result = json.loads(sub_route_text)
                except json.JSONDecodeError:
                    return f"SQL子路由返回的不是有效的JSON格式: {sub_route_text}"
            except Exception as e:
                return f"SQL子路由处理失败: {str(e)}"

            sub_type = sub_route_result.get("sub_type")
            if not sub_type:
                return f"无法解析子路由结果，缺少 'sub_type': {sub_route_result}"

            if sub_type == "db_structure":
                return self.handle_db_structure(inputs)
            elif sub_type == "sql_data":
                return self.handle_sql_data(inputs)
            else:
                return f"未知的sub_type: {sub_type}"

        elif top_type == "report":
            return self.handle_report(inputs)
        elif top_type == "visualization":
            return self.handle_visualization(inputs)
        elif top_type == "chat":
            return self.handle_chat(inputs)
        else:
            return f"未知的type: {top_type}"
    
    def handle_db_structure(self, inputs: dict) -> str:
        """处理查看/解释数据库结构的需求。"""
        print("正在获取数据库结构信息...", flush=True)
        partial_schema = get_partial_schema(self.db, max_tables=5, max_columns=10)
        
        print("正在分析数据库结构...", flush=True)
        
        # 构造提示消息
        messages = self.db_explain_prompt.format(
            question=inputs["question"],
            partial_schema=partial_schema
        )
        
        # 使用统一的流式输出处理方法
        return self._handle_streaming(messages, error_prefix="数据库结构分析")

    def handle_sql_data(self, inputs: dict) -> str:
        """处理真正要执行SQL查询的需求。"""
        print("正在分析您的需求...", flush=True)
        partial_schema = get_partial_schema(self.db, max_tables=5, max_columns=10)
        
        print("正在生成SQL查询...", flush=True)
        # 生成SQL查询
        messages = self.sql_prompt.format(
            partial_schema=partial_schema,
            question=inputs["question"]
        )
        sql_response = self._handle_streaming(messages, error_prefix="SQL生成")
        if "出错" in sql_response or "超时" in sql_response:
            return sql_response
            
        sql_cleaned = self.clean_sql_query(sql_response)

        print("正在执行SQL查询...", flush=True)
        result = run_query(self.db, sql_cleaned)
        
        # 先打印SQL和结果
        query_result = f"""执行的SQL查询: {sql_cleaned}
查询结果: {result}
---"""
        print(query_result, flush=True)
        
        print("正在分析查询结果...", flush=True)
        # 分析查询结果
        messages = self.result_explain_prompt.format(
            question=inputs["question"],
            sql=sql_cleaned,
            result=result
        )
        analysis = self._handle_streaming(messages, error_prefix="查询结果分析")
        
        # 返回完整内容用于历史记录
        return f"{query_result}{analysis}"

    def handle_report(self, inputs: dict) -> str:
        """处理生成分析报告的需求。"""
        print("正在整理历史数据...", flush=True)
        history_text = inputs.get("history", "")
        
        print("正在生成分析报告...", flush=True)
        messages = self.report_prompt.format(
            question=inputs["question"],
            history=history_text
        )
        return self._handle_streaming(messages, error_prefix="报告生成")

    def handle_visualization(self, inputs: dict) -> str:
        """处理数据可视化需求"""
        print("正在分析可视化需求...", flush=True)
        partial_schema = get_partial_schema(self.db, max_tables=5, max_columns=10)
        
        print("正在生成图表配置...", flush=True)
        messages = self.visualization_prompt.format(
            partial_schema=partial_schema,
            question=inputs["question"]
        )
        
        # 获取图表配置
        config_response = self._handle_streaming(messages, error_prefix="图表配置生成")
        try:
            config = json.loads(self._clean_json_response(config_response))
        except json.JSONDecodeError:
            return "生成的图表配置格式无效，请重试。"
        
        print("正在查询数据...", flush=True)
        try:
            data = run_query(self.db, config["sql"])
            if isinstance(data, str):  # 如果返回错误信息
                return f"查询数据时出错: {data}"
        except Exception as e:
            return f"执行查询时出错: {e}"
        
        # 返回特殊格式的响应，包含图表数据和配置
        return json.dumps({
            "type": "visualization",
            "data": data,
            "config": config
        })

    def handle_chat(self, inputs: dict) -> str:
        """处理普通对话或解释已有内容的需求"""
        print("正在思考回答...", flush=True)
        messages = self.chat_prompt.format_messages(
            history=inputs.get("history", ""),
            question=inputs["question"]
        )
        return self._handle_streaming(messages, error_prefix="思考")
    
    def chat(self, user_question: str, history: List[Tuple[str, str]] = None) -> str:
        """对话入口方法，外部只要调用 chat(问题文本) 即可。"""
        if history is None:
            history = []

        try:
            history_text = self._format_history(history)
            result = self.route_and_process({
                "question": user_question,
                "history": history_text
            })
            # 将最新问答加入历史
            history.append((user_question, result))
            return result

        except Exception as e:
            return f"抱歉，处理您的问题时出现错误: {e}"

    def _handle_streaming(self, messages, error_prefix: str = "处理") -> str:
        """统一处理流式输出和错误处理
        
        Args:
            messages: 提示消息
            error_prefix: 错误信息前缀
            
        Returns:
            str: 完整的响应内容
        """
        full_response = ""
        try:
            for chunk in self.llm.stream(messages):
                if chunk and hasattr(chunk, 'content') and chunk.content:
                    print(chunk.content, end='', flush=True)
                    full_response += chunk.content
            print()
            return full_response
            
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                return f"{error_prefix}超时，请稍后重试。"
            elif "connection" in error_msg.lower():
                return f"连接服务失败，请检查网络。"
            else:
                return f"{error_prefix}出错: {e}"

    def _format_history(self, history: List[Tuple[str, str]]) -> str:
        """格式化历史记录
        
        Args:
            history: 历史记录列表，每项是(问题, 回答)元组
            
        Returns:
            str: 格式化后的历史文本
        """
        formatted = []
        for q, a in history:
            formatted.extend([
                "用户问题：" + q,
                "助手回答：" + a,
                "---"
            ])
        return "\n".join(formatted)

    def _initialize_prompts(self):
        # SQL生成提示模板
        self.sql_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个SQL专家。请根据用户问题生成合适的SQL查询语句。

            规则：
            1. 只返回一条SQL语句，不要包含任何其他内容
            2. 不要使用markdown格式或代码块标记
            3. 不要添加注释或解释
            4. 确保SQL语法正确
            5. 对于探索性查询（如"找找看"、"看看有没有"等）：
               - 优先使用 INFORMATION_SCHEMA.COLUMNS 查找相关列
               - 对于销售相关查询，查找包含 'sale'、'invoice'、'order'、'amount'、'price'、'total' 等关键词的列
               - 对于用户相关查询，查找包含 'user'、'customer'、'client'、'person' 等关键词的列
               - 对于产品相关查询，查找包含 'product'、'item'、'goods'、'track'、'album' 等关键词的列
               - 返回找到的表和列的简要信息
            6. 对于分析类查询：
               - 适当使用聚合函数（SUM, AVG, COUNT等）
               - 考虑使用 GROUP BY 进行分组
               - 使用 ORDER BY 对结果排序
               - 可以使用子查询获取更复杂的统计信息
            
            这是数据库的一部分结构信息（可能被截断）：
            {partial_schema}
            """),
            ("human", """
            用户原始问题：{question}
            
            请生成合适的SQL查询：
            """)
        ])
        
        # 创建响应生成提示模板 - 用于解释SQL查询的结果
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
        
        # 路由提示模板
        self.router_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个对话分类专家。请根据用户输入判断查询类型：

            1. sql_query: 需要查询数据库的问题
               - 查看数据库结构（包含"讲讲"、"解释"、"说说"等词）
               - 查询具体数据（包含"查询"、"显示"、"列出"等词）
               - 统计和分析数据（包含"分析"、"比较"、"统计"等词）
               - 探索性查询（如"找找看"、"看看哪些"、"有没有"等词）
               - 涉及具体数据的比较或分析

            2. report: 需要生成分析报告的问题
               - 明确要求生成完整报表
               - 需要多维度综合分析
               - 明确包含"分析报告"、"总结报告"等词
               - 需要长篇幅的分析说明

            3. visualization: 需要生成可视化图表的问题
               - 明确要求生成图表
               - 包含"画图"、"展示图"等词
               - 指定了具体的图表类型

            4. chat: 普通对话
               - 日常问候
               - 系统相关问题
               - 不涉及数据分析的交谈

            规则：
            1. 优先选择 sql_query，除非明确要求其他类型
            2. 涉及数据分析时，应该先用 sql_query 获取数据
            3. 只有在明确要求完整报告时才选择 report
            4. 只有在明确要求图表时才选择 visualization

            严格按照JSON格式输出：{{"type": "sql_query"}} 或 {{"type":"report"}} 或 {{"type":"visualization"}} 或 {{"type":"chat"}}"""),
            ("human", "{question}")
        ])
        # =============== 第二层路由（仅当 type=sql_query 时） ===============
        # 判断是 db_structure 还是 sql_data
        self.sql_router_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个SQL需求分类专家。用户的问题属于sql_query大类，请判断具体类型：

            1. db_structure: 
               - 需要完整解释数据库结构
               - 需要详细说明表关系
               - 通常包含"结构"、"架构"、"设计"等词

            2. sql_data:
               - 需要执行SQL查询获取数据
               - 包含具体的数据查询需求
               - 探索性查询（如"找找"、"看看哪些"、"有没有"等）
               - 需要通过查询来了解数据内容
            
            请根据用户输入判定，并严格输出JSON：
            {{"sub_type": "db_structure"}} 或 {{"sub_type": "sql_data"}}
            """),
            ("human", "{question}")
        ])

        # SQL查询提示模板
        self.sql_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个SQL专家。请根据用户问题生成合适的SQL查询语句。

            规则：
            1. 只返回一条SQL语句，不要包含任何其他内容
            2. 不要使用markdown格式或代码块标记
            3. 不要添加注释或解释
            4. 确保SQL语法正确
            5. 对于探索性查询（如"找找看"、"看看有没有"等）：
               - 优先使用 INFORMATION_SCHEMA.COLUMNS 查找相关列
               - 对于销售相关查询，查找包含 'sale'、'invoice'、'order'、'amount'、'price'、'total' 等关键词的列
               - 对于用户相关查询，查找包含 'user'、'customer'、'client'、'person' 等关键词的列
               - 对于产品相关查询，查找包含 'product'、'item'、'goods'、'track'、'album' 等关键词的列
               - 返回找到的表和列的简要信息
            6. 对于分析类查询：
               - 适当使用聚合函数（SUM, AVG, COUNT等）
               - 考虑使用 GROUP BY 进行分组
               - 使用 ORDER BY 对结果排序
               - 可以使用子查询获取更复杂的统计信息
            
            这是数据库的一部分结构信息（可能被截断）：
            {partial_schema}
            """),
            ("human", """
            用户原始问题：{question}
            
            请生成合适的SQL查询：
            """)
        ])
        
        # 数据库解释提示模板
        self.db_explain_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个数据库专家，请解释数据库的结构。

            在回答时，请确保：
            - 先简要概述数据库的内容和用途
            - 清晰解释主要表的作用和它们之间的关系
            - 突出说明核心业务表
            - 使用通俗易懂的语言
            - 适时举例说明典型业务场景
            
            注意：不要按点列举回答，而是用流畅的语言自然地表达。
            """),
            ("human", """
            用户问题：{question}
            
            数据库信息(可能被截断)：
            {partial_schema}
            
            请给出解释：
            """)
        ])
        
        # 报表生成提示模板
        self.report_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的数据分析师。请根据用户问题和历史查询生成分析报告。

            在生成报告时，请确保：
            - 给出清晰的报告结构，包含概述和详细分析
            - 突出关键发现和数据趋势
            - 提供具体的业务建议
            - 使用专业但易懂的语言
            - 保持内容与用户问题的相关性
            
            注意：用自然的方式组织内容，避免机械的点式列举。
            """),
            ("human", """
            用户问题：{question}
            
            历史查询记录：
            {history}
            
            请生成分析报告：
            """)
        ])
        
        # SQL结果解释
        self.result_explain_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个数据分析师，请解释SQL查询结果。

            在分析结果时，请确保：
            - 用通俗易懂的语言解释数据含义
            - 指出重要的发现和数据模式
            - 突出说明任何趋势或特点
            - 对于探索性查询：
              • 说明找到的相关表和列的用途
              • 建议接下来可以进行哪些查询
              • 解释这些数据与用户需求的关联
            - 如果结果为空或有错误，解释原因并给出建议
            
            注意：用自然的语言流畅地表达，避免机械的点式列举。
            """),
            ("human", """
            用户原始问题：{question}
            执行的SQL查询：{sql}
            查询结果：{result}
            
            请分析这个结果：
            """)
        ])
        
        # 聊天提示模板
        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个友好的数据库助手。请根据上下文和用户问题进行回答。
            
            在回答时，请确保：
            - 保持友好专业的对话风格
            - 用通俗易懂的方式解释数据库概念
            - 在探索数据时，引导用户逐步了解数据结构
            - 提供具体的查询建议和方向
            - 解释相关的业务概念
            - 说明你可以帮助：
              • 解释数据库结构
              • 执行数据查询
              • 生成分析报告
              • 数据可视化
              • 回答一般性问题
            - 保持对话的连贯性，适当参考历史对话
            
            注意：像真实助手一样自然地交谈，避免机械的回答方式。
            
            历史对话：
            {history}
            """),
            ("human", """
            用户问题：{question}
            """)
        ])
        
        # 可视化提示模板
        self.visualization_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个数据可视化专家。请根据用户需求生成图表配置。

            在设计可视化时，请确保：
            - 选择最适合展示数据的图表类型
            - 生成准确的SQL查询获取所需数据
            - 设置清晰的标题和轴标签
            - 提供有见地的图表解释
            
            请以JSON格式返回配置（不要添加其他内容）：
            {
                "chart_type": "选择 bar/line/pie/scatter",
                "sql": "获取数据的SQL查询",
                "title": "图表标题",
                "x_label": "X轴标签",
                "y_label": "Y轴标签",
                "description": "图表解释"
            }
            """),
            ("human", """
            数据库结构：
            {partial_schema}
            
            用户需求：{question}
            """)
        ])
        
        # ========== 将 Prompt + LLM + Parser 组合起来 ==========
        # 只保留需要特殊解析的路由链
        self.router_chain = (
            self.router_prompt 
            | self.non_streaming_llm 
            | JsonOutputParser()
        )

        self.sql_sub_router_chain = (
            self.sql_router_prompt
            | self.non_streaming_llm
            | JsonOutputParser()
        )
        
        # 移除其他链，因为我们直接使用流式输出
        # self.sql_chain = (
        #     self.sql_prompt 
        #     | self.llm 
        #     | StrOutputParser()
        # )


def main():
    """主函数：创建聊天机器人实例并在命令行进行交互。"""
    max_retries = 3
    retry_delay = 5  # 秒
    
    for attempt in range(max_retries):
        try:
            chatbot = DBChatBot()
            if chatbot.llm:  # 如果LLM初始化成功
                break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"初始化失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                print(f"将在 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                print(f"无法初始化聊天机器人: {e}")
                return

    print("欢迎使用数据库聊天机器人！输入 'quit' 退出。")
    
    history = []

    while True:
        user_input = input("\n请输入您的问题: ")
        if user_input.lower() == 'quit':
            break
        
        response = chatbot.chat(user_input, history)
        print(f"\n{response}")


if __name__ == "__main__":
    main()