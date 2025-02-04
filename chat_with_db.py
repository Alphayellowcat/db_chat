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
from visualization import ChartGenerator
from report_generator import ReportGenerator

# 设置API密钥
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_schema(db) -> str:
    """获取完整的数据库schema信息，包括表结构、字段描述等
    
    Args:
        db: SQLDatabase实例
    Returns:
        str: 数据库详细结构信息
    """
    try:
        # 获取所有表的信息
        tables_query = """
        SELECT 
            t.TABLE_NAME,
            t.TABLE_COMMENT
        FROM information_schema.TABLES t
        WHERE t.TABLE_SCHEMA = DATABASE()
        """
        tables_info = db.run(tables_query)
        
        # 获取所有列的信息
        columns_query = """
        SELECT 
            c.TABLE_NAME,
            c.COLUMN_NAME,
            c.COLUMN_TYPE,
            c.IS_NULLABLE,
            c.COLUMN_DEFAULT,
            c.COLUMN_COMMENT,
            c.EXTRA
        FROM information_schema.COLUMNS c
        WHERE c.TABLE_SCHEMA = DATABASE()
        ORDER BY c.TABLE_NAME, c.ORDINAL_POSITION
        """
        columns_info = db.run(columns_query)
        
        # 获取外键关系
        fk_query = """
        SELECT 
            k.TABLE_NAME,
            k.COLUMN_NAME,
            k.REFERENCED_TABLE_NAME,
            k.REFERENCED_COLUMN_NAME
        FROM information_schema.KEY_COLUMN_USAGE k
        WHERE k.TABLE_SCHEMA = DATABASE()
        AND k.REFERENCED_TABLE_NAME IS NOT NULL
        """
        fk_info = db.run(fk_query)
        
        # 格式化输出
        schema_info = []
        schema_info.append("数据库结构信息：\n")
        
        # 处理表信息
        tables_dict = {}
        if isinstance(tables_info, str):
            tables_info = ast.literal_eval(tables_info)
        for table in tables_info:
            table_name = table[0]
            table_comment = table[1] or "无描述"
            tables_dict[table_name] = {
                "comment": table_comment,
                "columns": [],
                "foreign_keys": []
            }
            
        # 处理列信息
        if isinstance(columns_info, str):
            columns_info = ast.literal_eval(columns_info)
        for col in columns_info:
            table_name = col[0]
            if table_name in tables_dict:
                col_info = {
                    "name": col[1],
                    "type": col[2],
                    "nullable": col[3],
                    "default": col[4],
                    "comment": col[5] or "无描述",
                    "extra": col[6]
                }
                tables_dict[table_name]["columns"].append(col_info)
                
        # 处理外键信息
        if isinstance(fk_info, str):
            fk_info = ast.literal_eval(fk_info)
        for fk in fk_info:
            table_name = fk[0]
            if table_name in tables_dict:
                fk_info = {
                    "column": fk[1],
                    "referenced_table": fk[2],
                    "referenced_column": fk[3]
                }
                tables_dict[table_name]["foreign_keys"].append(fk_info)
        
        # 生成最终输出
        for table_name, table_info in tables_dict.items():
            schema_info.append(f"\n表名：{table_name}")
            schema_info.append(f"表描述：{table_info['comment']}")
            
            schema_info.append("\n字段列表：")
            for col in table_info["columns"]:
                nullable = "可空" if col["nullable"] == "YES" else "非空"
                default = f"默认值: {col['default']}" if col["default"] else "无默认值"
                schema_info.append(
                    f"- {col['name']} ({col['type']}) {nullable} {default}"
                    f"\n  描述: {col['comment']}"
                    f"\n  其他: {col['extra']}"
                )
            
            if table_info["foreign_keys"]:
                schema_info.append("\n外键关系：")
                for fk in table_info["foreign_keys"]:
                    schema_info.append(
                        f"- {fk['column']} -> "
                        f"{fk['referenced_table']}.{fk['referenced_column']}"
                    )
            
            schema_info.append("\n" + "="*50)
        
        return "\n".join(schema_info)
        
    except Exception as e:
        return f"获取数据库结构信息时出错: {str(e)}"

def run_query(db, query: str):
    """执行SQL查询
    Args:
        db: SQLDatabase实例
        query (str): SQL查询语句
    Returns:
        str: 查询结果或错误信息
    """
    try:
        print(f"执行SQL查询: {query}")  # 调试信息
        result = db.run(query)
        print(f"查询结果: {result}")  # 调试信息
        return result
    except Exception as e:
        import traceback
        print(f"SQL查询执行错误: {str(e)}")
        print(f"错误堆栈: {traceback.format_exc()}")
        return f"查询执行错误: {str(e)}"

def get_partial_schema(db, max_tables=5, max_columns=10) -> str:
    """获取部分数据库结构，避免Prompt过长
    
    Args:
        db: SQLDatabase实例
        max_tables: 最大返回表数量
        max_columns: 每表最大返回字段数量
    Returns:
        str: 部分数据库结构信息
    """
    try:
        # 获取最重要的表（基于外键关系数量和表注释）
        importance_query = """
        SELECT 
            t.TABLE_NAME,
            t.TABLE_COMMENT,
            COUNT(k.REFERENCED_TABLE_NAME) as reference_count
        FROM information_schema.TABLES t
        LEFT JOIN information_schema.KEY_COLUMN_USAGE k 
            ON t.TABLE_NAME = k.TABLE_NAME 
            AND k.TABLE_SCHEMA = DATABASE()
            AND k.REFERENCED_TABLE_NAME IS NOT NULL
        WHERE t.TABLE_SCHEMA = DATABASE()
        GROUP BY t.TABLE_NAME, t.TABLE_COMMENT
        ORDER BY reference_count DESC, 
                 CASE WHEN t.TABLE_COMMENT != '' THEN 0 ELSE 1 END,
                 t.TABLE_NAME
        LIMIT %d
        """ % max_tables
        
        print(f"执行表查询: {importance_query}")  # 调试信息
        tables_info = db.run(importance_query)
        print(f"表查询结果: {tables_info}")  # 调试信息
        
        if not tables_info:
            return "数据库中没有找到表"
            
        if isinstance(tables_info, str):
            try:
                tables_info = ast.literal_eval(tables_info)
            except Exception as e:
                print(f"解析表信息失败: {e}, 原始数据: {tables_info}")  # 调试信息
                return f"解析表信息失败: {str(e)}"
            
        schema_info = []
        schema_info.append("重要表结构信息：\n")
        
        if not isinstance(tables_info, (list, tuple)):
            return f"表信息格式错误: {type(tables_info)}"
            
        for table_info in tables_info:
            if not isinstance(table_info, (list, tuple)) or len(table_info) < 2:
                continue
                
            table_name = table_info[0]
            table_comment = table_info[1] or "无描述"
            
            # 获取表的列信息
            columns_query = """
            SELECT 
                COLUMN_NAME,
                COLUMN_TYPE,
                IS_NULLABLE,
                COLUMN_DEFAULT,
                COLUMN_COMMENT
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
            AND TABLE_NAME = '%s'
            ORDER BY ORDINAL_POSITION
            LIMIT %d
            """ % (table_name, max_columns)
            
            print(f"执行列查询: {columns_query}")  # 调试信息
            columns_info = db.run(columns_query)
            print(f"列查询结果: {columns_info}")  # 调试信息
            
            if isinstance(columns_info, str):
                try:
                    columns_info = ast.literal_eval(columns_info)
                except Exception as e:
                    print(f"解析列信息失败: {e}, 原始数据: {columns_info}")  # 调试信息
                    continue
            
            schema_info.append(f"\n表名：{table_name}")
            schema_info.append(f"表描述：{table_comment}")
            
            if not columns_info:
                schema_info.append("\n该表没有列信息")
                continue
                
            schema_info.append("\n重要字段：")
            for col in columns_info:
                if not isinstance(col, (list, tuple)) or len(col) < 5:
                    continue
                    
                nullable = "可空" if col[2] == "YES" else "非空"
                default = f"默认值: {col[3]}" if col[3] else "无默认值"
                comment = col[4] or "无描述"
                schema_info.append(
                    f"- {col[0]} ({col[1]}) {nullable} {default}"
                    f"\n  描述: {comment}"
                )
            
            # 获取外键关系
            fk_query = """
            SELECT 
                COLUMN_NAME,
                REFERENCED_TABLE_NAME,
                REFERENCED_COLUMN_NAME
            FROM information_schema.KEY_COLUMN_USAGE
            WHERE TABLE_SCHEMA = DATABASE()
            AND TABLE_NAME = '%s'
            AND REFERENCED_TABLE_NAME IS NOT NULL
            """ % table_name
            
            print(f"执行外键查询: {fk_query}")  # 调试信息
            fk_info = db.run(fk_query)
            print(f"外键查询结果: {fk_info}")  # 调试信息
            
            if isinstance(fk_info, str):
                try:
                    fk_info = ast.literal_eval(fk_info)
                except Exception as e:
                    print(f"解析外键信息失败: {e}, 原始数据: {fk_info}")  # 调试信息
                    continue
            
            if fk_info:
                schema_info.append("\n关联关系：")
                for fk in fk_info:
                    if not isinstance(fk, (list, tuple)) or len(fk) < 3:
                        continue
                    schema_info.append(
                        f"- {fk[0]} -> {fk[1]}.{fk[2]}"
                    )
            
            schema_info.append("\n" + "="*50)
        
        return "\n".join(schema_info)
        
    except Exception as e:
        import traceback
        print(f"获取部分数据库结构信息时出错: {str(e)}")
        print(f"错误堆栈: {traceback.format_exc()}")  # 打印完整错误堆栈
        return f"获取部分数据库结构信息时出错: {str(e)}"


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
        self.chart_generator = ChartGenerator()  # 初始化图表生成器
        self.report_generator = ReportGenerator()  # 初始化报告生成器
        
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
        try:
            print("正在获取数据库结构信息...", flush=True)
            partial_schema = get_partial_schema(self.db, max_tables=5, max_columns=10)
            
            if "错误" in partial_schema:
                return f"获取数据库结构失败：{partial_schema}"
            
            print("正在分析数据库结构...", flush=True)
            
            # 构造提示消息
            messages = self.db_explain_prompt.format(
                question=inputs["question"],
                partial_schema=partial_schema
            )
            
            # 使用统一的流式输出处理方法
            return self._handle_streaming(messages, error_prefix="数据库结构分析")
        except Exception as e:
            import traceback
            print(f"处理数据库结构请求时出错: {str(e)}")
            print(f"错误堆栈: {traceback.format_exc()}")
            return f"处理数据库结构请求时出错: {str(e)}"

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
        query_result = sql_cleaned
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
        return analysis

    def handle_report(self, inputs: dict) -> str:
        """处理生成分析报告的需求"""
        print("正在分析报告需求...", flush=True)
        
        # 获取数据库结构信息
        partial_schema = get_partial_schema(self.db, max_tables=5, max_columns=10)
        
        # 生成报告大纲
        print("正在生成报告大纲...", flush=True)
        outline_messages = self.report_outline_prompt.format(
            question=inputs["question"],
            schema=partial_schema,
            history=inputs.get("history", "")
        )
        
        outline_response = self._handle_streaming(outline_messages, error_prefix="报告大纲生成")
        
        try:
            # 解析大纲响应
            outline = json.loads(outline_response)
            required_fields = ["title", "sections", "queries"]
            for field in required_fields:
                if field not in outline:
                    raise ValueError(f"大纲缺少必要字段: {field}")
                    
        except json.JSONDecodeError as e:
            return f"生成的报告大纲格式无效: {str(e)}\n原始响应: {outline_response}"
        except ValueError as e:
            return f"报告大纲验证失败: {str(e)}"
        except Exception as e:
            return f"处理报告大纲时出错: {str(e)}"
            
        # 执行数据查询
        print("正在收集数据...", flush=True)
        charts_data = []
        query_results = []
        
        for query_info in outline["queries"]:
            try:
                # 执行查询
                data = run_query(self.db, query_info["sql"])
                if isinstance(data, str):
                    data = ast.literal_eval(data)
                    
                # 如果需要可视化
                if query_info.get("visualization"):
                    vis_config = query_info["visualization"]
                    charts_data.append({
                        "data": data,
                        "config": vis_config
                    })
                    
                # 保存查询结果
                query_results.append({
                    "name": query_info["name"],
                    "data": data
                })
                
            except Exception as e:
                print(f"执行查询 {query_info['name']} 时出错: {str(e)}")
                continue
                
        # 生成报告内容
        print("正在生成报告内容...", flush=True)
        content_messages = self.report_content_prompt.format(
            question=inputs["question"],
            outline=json.dumps(outline, ensure_ascii=False),
            results=json.dumps(query_results, ensure_ascii=False)
        )
        
        content_response = self._handle_streaming(content_messages, error_prefix="报告内容生成")
        
        # 生成完整报告
        print("正在生成完整报告...", flush=True)
        try:
            report_info = self.report_generator.generate_report(
                title=outline["title"],
                content=content_response,
                charts_data=charts_data if charts_data else None
            )
            
            return json.dumps({
                "type": "report",
                "data": {
                    "format": report_info["format"],
                    "path": report_info["path"],
                    "title": outline["title"]
                }
            })
            
        except Exception as e:
            return f"生成报告时出错: {str(e)}"

    def handle_visualization(self, inputs: dict) -> str:
        """处理数据可视化需求"""
        try:
            print("正在分析可视化需求...", flush=True)
            partial_schema = get_partial_schema(self.db, max_tables=5, max_columns=10)
            
            if "错误" in partial_schema:
                return f"获取数据库结构失败：{partial_schema}"
            
            print("正在生成图表配置...", flush=True)
            messages = self.visualization_prompt.format(
                partial_schema=partial_schema,
                question=inputs["question"]
            )
            
            # 获取图表配置
            config_response = self._handle_streaming(messages, error_prefix="图表配置生成")
            if "错误" in config_response or "超时" in config_response:
                return config_response
            
            try:
                # 清理响应文本
                cleaned_response = config_response.strip()
                print(f"原始配置响应: {cleaned_response}")  # 调试信息
                
                # 尝试解析 JSON
                try:
                    config = json.loads(cleaned_response)
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {e}, 原始文本: {cleaned_response}")
                    return f"生成的图表配置格式无效: {str(e)}"
                
                # 验证必要的字段
                required_fields = ["chart_type", "sql", "title", "x_label", "y_label", "description"]
                missing_fields = [field for field in required_fields if field not in config]
                if missing_fields:
                    return f"图表配置缺少必要字段: {', '.join(missing_fields)}"
                
                # 验证图表类型
                valid_chart_types = ["bar", "line", "pie", "scatter"]
                if config["chart_type"] not in valid_chart_types:
                    return f"不支持的图表类型: {config['chart_type']}"
                
                print(f"验证后的配置: {json.dumps(config, ensure_ascii=False)}")  # 调试信息
                
            except Exception as e:
                print(f"处理配置时出错: {str(e)}")
                return f"处理图表配置时出错: {str(e)}"
            
            print("正在查询数据...", flush=True)
            try:
                # 执行SQL查询
                data = run_query(self.db, config["sql"])
                print(f"查询结果: {data}")  # 调试信息
                
                if isinstance(data, str):
                    if "错误" in data:
                        return f"查询数据时出错: {data}"
                    try:
                        data = ast.literal_eval(data)
                    except Exception as e:
                        return f"解析查询结果失败: {str(e)}"
                
                if not data:
                    return "查询结果为空，无法生成图表"
                    
            except Exception as e:
                print(f"执行查询时出错: {str(e)}")
                return f"执行查询时出错: {str(e)}"
            
            print("正在生成图表...", flush=True)
            try:
                # 生成图表
                chart_info = self.chart_generator.generate_chart(data, config)
                print(f"图表信息: {json.dumps(chart_info, ensure_ascii=False)}")  # 调试信息
                
                # 检查文件是否生成
                if not os.path.exists(chart_info["static_path"]):
                    return "静态图表文件生成失败"
                if not os.path.exists(chart_info["interactive_path"]):
                    return "交互式图表文件生成失败"
                
                # 返回结果
                result = {
                    "type": "visualization",
                    "data": {
                        "static_chart": chart_info["static_path"],
                        "interactive_chart": chart_info["interactive_path"],
                        "title": chart_info["title"],
                        "description": chart_info["description"]
                    }
                }
                
                return json.dumps(result, ensure_ascii=False)
                
            except Exception as e:
                print(f"生成图表时出错: {str(e)}")
                return f"生成图表时出错: {str(e)}"
            
        except Exception as e:
            print(f"处理可视化请求时出错: {str(e)}")
            return f"处理可视化请求时出错: {str(e)}"

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

            请严格按照JSON格式返回：{{"type": "sql_query"}} 或 {{"type":"report"}} 或 {{"type":"visualization"}} 或 {{"type":"chat"}}"""),
            ("human", "{question}")
        ])

        # SQL子路由提示模板
        self.sql_router_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个SQL需求分类专家。用户的问题属于sql_query大类，请判断具体类型：

            1. db_structure: 
               - 需要了解数据库结构
               - 需要解释表的关系
               - 包含"结构"、"表"、"字段"、"解释"等词

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

        # SQL生成提示模板
        self.sql_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个SQL专家。请根据用户问题生成合适的SQL查询语句。

            规则：
            1. 只返回一条SQL语句，不要包含任何其他内容
            2. 不要使用markdown格式或代码块标记
            3. 不要添加注释或解释
            4. 确保SQL语法正确
            5. 车型的字段是"VehicleType"，客车的值是"1"，货车的值是"2"
            6. 收费的字段是"TotalToll" 在出口表中
            7. 对于分析类查询：
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

            目标：
            1.先简要概述目标数据库的内容
            2.清晰解释主要表的作用和它们之间的关系
            3.突出说明核心业务表
            
            注意：按点列举回答，不要总结
            """),
            ("human", """
            用户问题：{question}
            
            数据库信息(可能被截断)：
            {partial_schema}
            
            请给出解释：
            """)
        ])
        
        # 报告大纲生成提示模板
        self.report_outline_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的数据分析师。请根据用户需求和数据库结构生成报告大纲。

            请严格按照以下JSON格式返回大纲：
            {
                "title": "报告标题",
                "sections": [
                    {
                        "title": "章节标题",
                        "content_type": "text|chart|table",
                        "description": "本章节的分析重点"
                    }
                ],
                "queries": [
                    {
                        "name": "查询名称",
                        "sql": "SQL查询语句",
                        "purpose": "查询目的说明",
                        "visualization": {  # 可选，如果需要可视化
                            "chart_type": "bar|line|pie|scatter",
                            "title": "图表标题",
                            "x_label": "X轴标签",
                            "y_label": "Y轴标签",
                            "description": "图表说明"
                        }
                    }
                ]
            }
            
            注意：
            1. 确保生成有效的JSON
            2. queries中的SQL必须是可执行的
            3. 合理安排章节顺序
            4. 适当使用可视化
            """),
            ("human", """
            数据库结构：
            {schema}
            
            历史记录：
            {history}
            
            用户需求：{question}
            """)
        ])
        
        # 报告内容生成提示模板
        self.report_content_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的数据分析师。请根据报告大纲和查询结果生成详细的报告内容。

            要求：
            1. 内容要专业、客观
            2. 数据分析要有深度
            3. 结论要有理有据
            4. 建议要具体可行
            5. 使用Markdown格式
            6. 适当引用数据支持论点
            7. 突出重要发现
            8. 使用图表辅助说明
            """),
            ("human", """
            用户需求：{question}
            
            报告大纲：
            {outline}
            
            查询结果：
            {results}
            
            请生成报告内容：
            """)
        ])
        
        # SQL结果解释
        self.result_explain_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个数据分析师，请解释SQL查询结果。

            在分析结果时，请确保：
            - 用通俗易懂的语言解释数据含义

            
            注意：避免避免总结。
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

            你需要生成一个JSON格式的配置，包含以下必要字段：
            {
                "chart_type": "图表类型，必须是 'bar'、'line'、'pie' 或 'scatter' 之一",
                "sql": "用于获取数据的SQL查询语句",
                "title": "图表标题",
                "x_label": "X轴标签（对应SQL查询结果的第一列）",
                "y_label": "Y轴标签（对应SQL查询结果的第二列）",
                "description": "图表的详细说明"
            }

            注意事项：
            1. SQL查询必须返回至少两列数据：
               - 第一列用于X轴（分类、时间等）
               - 第二列用于Y轴（数值）
            2. 对于不同图表类型：
               - bar（柱状图）：适用于分类比较
               - line（折线图）：适用于趋势分析
               - pie（饼图）：适用于占比分析
               - scatter（散点图）：适用于相关性分析
            3. 确保SQL查询是有效的，并且字段名正确
            4. 标题和说明应该清晰描述图表内容
            5. 不要添加任何注释或额外内容
            6. 确保生成的是有效的JSON字符串

            数据库结构信息：
            {partial_schema}
            """),
            ("human", """
            用户需求：{question}
            
            请生成图表配置：
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