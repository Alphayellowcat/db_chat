import os
import time
from typing import Dict, Any, List, Union
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ChartGenerator:
    def __init__(self, output_dir: str = "charts"):
        """初始化图表生成器
        
        Args:
            output_dir: 图表输出目录
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置Seaborn样式
        sns.set_style("whitegrid")
        
    def generate_chart(self, data: List[tuple], config: Dict[str, Any]) -> Dict[str, Any]:
        """生成图表
        
        Args:
            data: 查询结果数据
            config: 图表配置
            
        Returns:
            Dict: 包含图表信息的字典
        """
        try:
            # 转换数据为DataFrame
            df = pd.DataFrame(data)
            if len(df.columns) >= 2:  # 确保至少有x和y轴数据
                df.columns = [config.get('x_label', 'x'), config.get('y_label', 'y')] + \
                           [f'col_{i}' for i in range(len(df.columns)-2)]
            
            # 生成静态图表（使用matplotlib）
            static_path = self._generate_static_chart(df, config)
            
            # 生成交互式图表（使用plotly）
            interactive_path = self._generate_interactive_chart(df, config)
            
            return {
                "static_path": static_path,
                "interactive_path": interactive_path,
                "title": config["title"],
                "description": config["description"]
            }
            
        except Exception as e:
            raise Exception(f"生成图表时出错: {str(e)}")
    
    def _generate_static_chart(self, df: pd.DataFrame, config: Dict[str, Any]) -> str:
        """生成静态图表
        
        Args:
            df: 数据框
            config: 图表配置
            
        Returns:
            str: 图表文件路径
        """
        plt.figure(figsize=(10, 6))
        
        chart_type = config["chart_type"].lower()
        if chart_type == "bar":
            plt.bar(df[config["x_label"]], df[config["y_label"]])
        elif chart_type == "line":
            plt.plot(df[config["x_label"]], df[config["y_label"]], marker='o')
        elif chart_type == "pie":
            plt.pie(df[config["y_label"]], labels=df[config["x_label"]], autopct='%1.1f%%')
        elif chart_type == "scatter":
            plt.scatter(df[config["x_label"]], df[config["y_label"]])
        else:
            raise ValueError(f"不支持的图表类型: {chart_type}")
        
        plt.title(config["title"])
        plt.xlabel(config["x_label"])
        plt.ylabel(config["y_label"])
        
        # 如果数据点过多，旋转x轴标签
        if len(df) > 10:
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        filename = f"static_{int(time.time())}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _generate_interactive_chart(self, df: pd.DataFrame, config: Dict[str, Any]) -> str:
        """生成交互式图表
        
        Args:
            df: 数据框
            config: 图表配置
            
        Returns:
            str: 图表HTML文件路径
        """
        chart_type = config["chart_type"].lower()
        
        if chart_type == "bar":
            fig = px.bar(df, x=config["x_label"], y=config["y_label"],
                        title=config["title"])
        elif chart_type == "line":
            fig = px.line(df, x=config["x_label"], y=config["y_label"],
                         title=config["title"], markers=True)
        elif chart_type == "pie":
            fig = px.pie(df, values=config["y_label"], names=config["x_label"],
                        title=config["title"])
        elif chart_type == "scatter":
            fig = px.scatter(df, x=config["x_label"], y=config["y_label"],
                           title=config["title"])
        else:
            raise ValueError(f"不支持的图表类型: {chart_type}")
        
        # 更新布局
        fig.update_layout(
            title_x=0.5,
            xaxis_title=config["x_label"],
            yaxis_title=config["y_label"],
            template="plotly_white"
        )
        
        # 保存为HTML文件
        filename = f"interactive_{int(time.time())}.html"
        filepath = os.path.join(self.output_dir, filename)
        fig.write_html(filepath)
        
        return filepath
        
    def generate_dashboard(self, charts_data: List[Dict[str, Any]]) -> str:
        """生成包含多个图表的仪表板
        
        Args:
            charts_data: 图表数据和配置列表
            
        Returns:
            str: 仪表板HTML文件路径
        """
        # 创建子图
        n_charts = len(charts_data)
        rows = (n_charts + 1) // 2  # 每行最多2个图表
        fig = make_subplots(rows=rows, cols=2, subplot_titles=[d["config"]["title"] for d in charts_data])
        
        for i, chart_data in enumerate(charts_data):
            row = i // 2 + 1
            col = i % 2 + 1
            
            df = pd.DataFrame(chart_data["data"])
            config = chart_data["config"]
            chart_type = config["chart_type"].lower()
            
            if chart_type == "bar":
                trace = go.Bar(x=df[config["x_label"]], y=df[config["y_label"]])
            elif chart_type == "line":
                trace = go.Scatter(x=df[config["x_label"]], y=df[config["y_label"]], mode='lines+markers')
            elif chart_type == "pie":
                trace = go.Pie(values=df[config["y_label"]], labels=df[config["x_label"]])
            elif chart_type == "scatter":
                trace = go.Scatter(x=df[config["x_label"]], y=df[config["y_label"]], mode='markers')
            else:
                continue
                
            fig.add_trace(trace, row=row, col=col)
            
            # 更新轴标签
            fig.update_xaxes(title_text=config["x_label"], row=row, col=col)
            fig.update_yaxes(title_text=config["y_label"], row=row, col=col)
        
        # 更新布局
        fig.update_layout(
            height=400 * rows,
            width=1000,
            title_text="数据分析仪表板",
            showlegend=False,
            template="plotly_white"
        )
        
        # 保存仪表板
        filename = f"dashboard_{int(time.time())}.html"
        filepath = os.path.join(self.output_dir, filename)
        fig.write_html(filepath)
        
        return filepath 