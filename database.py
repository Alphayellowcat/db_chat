import mysql.connector
from config import MYSQL_CONFIG
from typing import List, Dict

class DatabaseManager:
    def __init__(self):
        self.connection = mysql.connector.connect(**MYSQL_CONFIG)
        self.cursor = self.connection.cursor(dictionary=True)
    
    def execute_query(self, query: str) -> List[Dict]:
        """执行SQL查询并返回结果"""
        try:
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            return results
        except Exception as e:
            print(f"查询执行错误: {e}")
            return []
    
    def get_table_schema(self, table_name: str) -> str:
        """获取表结构信息"""
        try:
            self.cursor.execute(f"DESCRIBE {table_name}")
            columns = self.cursor.fetchall()
            schema = f"表 {table_name} 的结构:\n"
            for col in columns:
                schema += f"- {col['Field']} ({col['Type']})\n"
            return schema
        except Exception as e:
            return f"获取表结构失败: {e}"
    
    def close(self):
        """关闭数据库连接"""
        self.cursor.close()
        self.connection.close() 