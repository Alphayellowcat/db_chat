import streamlit as st
import json
import pandas as pd
import ast
from chat_with_db import DBChatBot
import os

def initialize_session_state():
    """初始化 session state"""
    if "api_type" not in st.session_state:
        st.session_state["api_type"] = "deepseek"
        # st.session_state["api_type"] = "openai"

    if "chatbot" not in st.session_state:
        try:
            st.session_state["chatbot"] = DBChatBot(api_type=st.session_state["api_type"])
            st.session_state["api_connected"] = True
        except Exception as e:
            st.session_state["chatbot"] = None
            st.session_state["api_connected"] = False
            st.session_state["api_error"] = str(e)
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "processing" not in st.session_state:
        st.session_state["processing"] = False

def display_visualization(response_text):
    """显示可视化图表
    
    Args:
        response_text: JSON格式的响应文本，包含图表信息
    """
    try:
        # 解析响应
        response = json.loads(response_text)
        if response.get("type") != "visualization":
            raise ValueError("响应类型不是visualization")
            
        viz_data = response["data"]
        
        # 创建两列布局
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("静态图表")
            # 显示静态图表
            if os.path.exists(viz_data["static_chart"]):
                st.image(viz_data["static_chart"], caption=viz_data["title"])
            else:
                st.error("静态图表文件不存在")
        
        with col2:
            st.subheader("交互式图表")
            # 显示交互式图表
            if os.path.exists(viz_data["interactive_chart"]):
                with open(viz_data["interactive_chart"], 'r', encoding='utf-8') as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=400)
            else:
                st.error("交互式图表文件不存在")
        
        # 显示图表描述
        st.markdown("---")
        st.markdown(f"**图表说明：**\n{viz_data['description']}")
        
    except json.JSONDecodeError:
        st.error("解析图表数据失败：响应不是有效的JSON格式")
    except KeyError as e:
        st.error(f"图表数据缺少必要字段: {e}")
    except Exception as e:
        st.error(f"显示图表时出错: {e}")

def display_report(response_data: dict):
    """显示分析报告
    
    Args:
        response_data: 包含报告信息的字典
    """
    try:
        report_data = response_data["data"]
        report_format = report_data["format"]
        report_path = report_data["title"]
        
        st.subheader(f"📊 {report_data['title']}")
        
        if report_format == "markdown":
            if os.path.exists(report_path):
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_content = f.read()
                st.markdown(report_content)
            else:
                st.error("报告文件不存在")
                
        elif report_format == "pdf":
            if os.path.exists(report_path):
                with open(report_path, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    label="📥 下载PDF报告",
                    data=pdf_bytes,
                    file_name=os.path.basename(report_path),
                    mime="application/pdf"
                )
                # 显示PDF预览（如果环境支持）
                try:
                    st.components.v1.iframe(report_path, height=600)
                except:
                    st.info("请下载PDF文件查看完整报告")
            else:
                st.error("报告文件不存在")
        else:
            st.error(f"不支持的报告格式: {report_format}")
            
    except KeyError as e:
        st.error(f"报告数据缺少必要字段: {e}")
    except Exception as e:
        st.error(f"显示报告时出错: {e}")

def main():
    st.set_page_config(
        page_title="数据库助手",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    initialize_session_state()
    
    with st.sidebar:
        st.title("🤖 数据库助手")
        api_type = st.radio("选择 API 服务", ["OpenAI", "Deepseek"],
                           index=0 if st.session_state["api_type"] == "openai" else 1,
                           key="api_selector")
        if api_type.lower() != st.session_state["api_type"]:
            st.session_state["api_type"] = api_type.lower()
            try:
                st.session_state["chatbot"] = DBChatBot(api_type=st.session_state["api_type"])
                st.session_state["api_connected"] = True
                del st.session_state["api_error"]
                st.rerun()
            except Exception as e:
                st.session_state["api_connected"] = False
                st.session_state["api_error"] = str(e)
                st.error(f"API 切换失败: {e}")
        
        if st.session_state["api_connected"]:
            st.success(f"✅ {api_type} API 连接正常")
        else:
            st.error(f"❌ {api_type} API 连接失败")
            st.error(f"错误信息: {st.session_state['api_error']}")

        st.markdown("""
        ### 功能介绍
        1. 📊 查询数据库
        2. 📝 生成分析报告
        3. 📈 数据可视化
        4. 💬 解答问题
        
        ### 使用提示
        - 可以询问数据库结构
        - 可以查询具体数据
        - 可以请求生成分析报告
        - 可以要求生成图表
        - 可以进行普通对话
        """)
        
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state["history"] = []
            st.rerun()


    
    st.title("与数据库对话 💬")
    chat_container = st.container()
    
    with chat_container:
        st.session_state["rendered_messages"] = set()
        for q, a in st.session_state["history"]:
            if q not in st.session_state["rendered_messages"]:
                with st.chat_message("user", avatar="🧑"):
                    st.write(q)
                with st.chat_message("assistant", avatar="🤖"):
                    try:
                        # 尝试解析为JSON
                        response_data = json.loads(a)
                        if isinstance(response_data, dict):
                            if response_data.get("type") == "visualization":
                                display_visualization(a)
                            elif response_data.get("type") == "report":
                                display_report(response_data)
                            else:
                                st.write(a)
                        else:
                            st.write(a)
                    except json.JSONDecodeError:
                        st.write(a)
                st.session_state["rendered_messages"].add(q)
    
    if prompt := st.chat_input("请输入您的问题", disabled=not st.session_state["api_connected"]):
        if not st.session_state["processing"]:
            st.session_state["processing"] = True
            try:
                response = st.session_state["chatbot"].chat(prompt, st.session_state["history"])
                if not response.startswith("抱歉") and not response.startswith("错误"):
                    st.session_state["history"].append((prompt, response))
            except Exception as e:
                st.error(f"处理请求时出错: {e}")
                if "connection" in str(e).lower():
                    st.session_state["api_connected"] = False
                    st.session_state["api_error"] = str(e)
                    st.error("API 连接已断开，请在侧边栏重试连接")
            st.session_state["processing"] = False
            st.rerun()

if __name__ == "__main__":
    main()
