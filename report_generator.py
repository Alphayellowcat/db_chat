import os
import time
from typing import Dict, Any, List, Optional
import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from visualization import ChartGenerator

class ReportGenerator:
    def __init__(self, output_dir: str = "reports"):
        """初始化报告生成器
        
        Args:
            output_dir: 报告输出目录
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self.chart_generator = ChartGenerator()
        
    def generate_report(self, 
                       title: str,
                       content: str,
                       charts_data: Optional[List[Dict[str, Any]]] = None,
                       format: str = "markdown") -> Dict[str, str]:
        """生成报告
        
        Args:
            title: 报告标题
            content: 报告内容
            charts_data: 图表数据列表
            format: 输出格式 ("markdown" 或 "pdf")
            
        Returns:
            Dict: 包含报告文件路径的字典
        """
        try:
            # 生成报告基本内容
            report_content = self._generate_base_content(title, content)
            
            # 如果有图表数据，生成图表
            if charts_data:
                chart_sections = self._generate_chart_sections(charts_data)
                report_content += "\n\n" + chart_sections
            
            # 根据格式保存报告
            if format.lower() == "markdown":
                report_path = self._save_markdown(title, report_content)
                return {"format": "markdown", "path": report_path}
            elif format.lower() == "pdf":
                report_path = self._save_pdf(title, report_content)
                return {"format": "pdf", "path": report_path}
            else:
                raise ValueError(f"不支持的报告格式: {format}")
                
        except Exception as e:
            raise Exception(f"生成报告时出错: {str(e)}")
    
    def _generate_base_content(self, title: str, content: str) -> str:
        """生成报告基本内容
        
        Args:
            title: 报告标题
            content: 报告主体内容
            
        Returns:
            str: 格式化的报告内容
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        sections = [
            f"# {title}",
            f"\n生成时间：{now}\n",
            "## 目录",
            "1. 摘要",
            "2. 详细分析",
            "3. 数据可视化",
            "4. 结论与建议",
            "\n## 1. 摘要\n",
        ]
        
        # 分析内容中的章节
        content_parts = content.split("\n")
        current_section = []
        sections.extend(content_parts)
        
        return "\n".join(sections)
    
    def _generate_chart_sections(self, charts_data: List[Dict[str, Any]]) -> str:
        """生成图表部分
        
        Args:
            charts_data: 图表数据列表
            
        Returns:
            str: 图表部分的Markdown内容
        """
        sections = ["\n## 3. 数据可视化\n"]
        
        for i, chart_data in enumerate(charts_data, 1):
            try:
                # 生成图表
                chart_info = self.chart_generator.generate_chart(
                    chart_data["data"],
                    chart_data["config"]
                )
                
                # 添加图表说明
                sections.extend([
                    f"### 图表 {i}: {chart_info['title']}",
                    "",
                    f"![{chart_info['title']}]({chart_info['static_path']})",
                    "",
                    f"**说明**: {chart_info['description']}",
                    "",
                    f"[查看交互式图表]({chart_info['interactive_path']})",
                    ""
                ])
                
            except Exception as e:
                sections.extend([
                    f"### 图表 {i}",
                    "",
                    f"生成图表时出错: {str(e)}",
                    ""
                ])
        
        return "\n".join(sections)
    
    def _save_markdown(self, title: str, content: str) -> str:
        """保存为Markdown文件
        
        Args:
            title: 报告标题
            content: 报告内容
            
        Returns:
            str: 文件路径
        """
        filename = f"report_{int(time.time())}.md"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
            
        return filepath
    
    def _save_pdf(self, title: str, content: str) -> str:
        """保存为PDF文件
        
        Args:
            title: 报告标题
            content: 报告内容
            
        Returns:
            str: 文件路径
        """
        try:
            import markdown
            import pdfkit
            
            # 首先转换为HTML
            html = markdown.markdown(content)
            
            # 添加样式
            styled_html = f"""
            <html>
            <head>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1 {{ color: #2c3e50; }}
                    h2 {{ color: #34495e; margin-top: 30px; }}
                    h3 {{ color: #7f8c8d; }}
                    img {{ max-width: 100%; height: auto; }}
                    p {{ line-height: 1.6; }}
                </style>
            </head>
            <body>
                {html}
            </body>
            </html>
            """
            
            # 保存为PDF
            filename = f"report_{int(time.time())}.pdf"
            filepath = os.path.join(self.output_dir, filename)
            
            pdfkit.from_string(styled_html, filepath)
            
            return filepath
            
        except ImportError:
            raise Exception("生成PDF需要安装额外的依赖：pip install markdown pdfkit")
        except Exception as e:
            raise Exception(f"生成PDF时出错: {str(e)}")
            
    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析数据框，生成统计信息
        
        Args:
            df: 数据框
            
        Returns:
            Dict: 统计信息
        """
        analysis = {
            "basic_stats": {},
            "correlations": None,
            "missing_values": {},
            "unique_values": {}
        }
        
        try:
            # 基本统计信息
            analysis["basic_stats"] = df.describe().to_dict()
            
            # 相关性分析（仅对数值列）
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) >= 2:
                analysis["correlations"] = df[numeric_cols].corr().to_dict()
            
            # 缺失值统计
            analysis["missing_values"] = df.isnull().sum().to_dict()
            
            # 唯一值统计
            for col in df.columns:
                analysis["unique_values"][col] = df[col].nunique()
            
        except Exception as e:
            print(f"数据分析时出错: {str(e)}")
            
        return analysis 