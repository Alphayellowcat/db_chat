# app.py
# 在终端中运行：streamlit run db_chat/app.py
import streamlit as st
import json
import pandas as pd
import ast
from chat_with_db import DBChatBot

def initialize_session_state():
    """初始化session state"""
    if "api_type" not in st.session_state:
        st.session_state["api_type"] = "openai"  # 默认使用 OpenAI
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
    """显示可视化图表"""
    try:
        response = json.loads(response_text)
        if not isinstance(response["data"], pd.DataFrame):
            # 将字符串形式的查询结果转换为DataFrame
            data_list = ast.literal_eval(response["data"])
            df = pd.DataFrame(data_list)
        else:
            df = response["data"]
        
        config = response["config"]
        
        # 根据图表类型显示不同的图表
        if config["chart_type"] == "bar":
            st.bar_chart(df, x=config.get("x_label"), y=config.get("y_label"))
        elif config["chart_type"] == "line":
            st.line_chart(df, x=config.get("x_label"), y=config.get("y_label"))
        elif config["chart_type"] == "pie":
            st.pie_chart(df)
        elif config["chart_type"] == "scatter":
            st.scatter_chart(df, x=config.get("x_label"), y=config.get("y_label"))
        
        # 显示图表说明
        st.markdown(f"**图表说明：**\n{config['description']}")
        
    except Exception as e:
        st.error(f"显示图表时出错: {e}")

def main():
    st.set_page_config(
        page_title="数据库助手",
        page_icon="🤖",
        layout="wide"
    )
    
    # 初始化session state
    initialize_session_state()
    
    # 侧边栏
    with st.sidebar:
        st.title("🤖 数据库助手")
        
        # API 选择
        api_type = st.radio(
            "选择 API 服务",
            ["OpenAI", "Deepseek"],
            index=0 if st.session_state["api_type"] == "openai" else 1,
            key="api_selector"
        )
        
        # 如果切换了 API
        if (api_type == "OpenAI" and st.session_state["api_type"] != "openai") or \
           (api_type == "Deepseek" and st.session_state["api_type"] != "deepseek"):
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
        
        # 显示API连接状态
        if st.session_state["api_connected"]:
            st.success(f"✅ {api_type} API 连接正常")
        else:
            st.error(f"❌ {api_type} API 连接失败")
            st.error(f"错误信息: {st.session_state['api_error']}")
            if st.button("🔄 重试连接"):
                try:
                    st.session_state["chatbot"] = DBChatBot(api_type=st.session_state["api_type"])
                    st.session_state["api_connected"] = True
                    del st.session_state["api_error"]
                    st.rerun()
                except Exception as e:
                    st.session_state["api_error"] = str(e)
                    st.error(f"重连失败: {e}")
        
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
        
        # 清空对话按钮
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state["history"] = []
            st.rerun()
    
    # 主界面
    st.title("与数据库对话 💬")
    
    # 显示对话历史
    chat_container = st.container()
    with chat_container:
        for i, (q, a) in enumerate(st.session_state["history"]):
            # 用户问题
            with st.chat_message("user", avatar="🧑"):
                st.write(q)
            # 助手回答
            with st.chat_message("assistant", avatar="🤖"):
                try:
                    response_data = json.loads(a)
                    if isinstance(response_data, dict) and response_data.get("type") == "visualization":
                        display_visualization(a)
                    else:
                        st.write(a)
                except json.JSONDecodeError:
                    st.write(a)
    
    # 用户输入
    if prompt := st.chat_input("请输入您的问题", disabled=not st.session_state["api_connected"]):
        if not st.session_state["processing"]:
            # 设置处理标志
            st.session_state["processing"] = True
            
            # 显示用户问题
            with st.chat_message("user", avatar="🧑"):
                st.write(prompt)
            
            # 显示助手回答
            with st.chat_message("assistant", avatar="🤖"):
                try:
                    response = st.session_state["chatbot"].chat(prompt, st.session_state["history"])
                    # 检查是否是可视化响应
                    try:
                        response_data = json.loads(response)
                        if isinstance(response_data, dict) and response_data.get("type") == "visualization":
                            display_visualization(response)
                        else:
                            st.write(response)
                    except json.JSONDecodeError:
                        st.write(response)
                    
                    # 只有在成功时才将对话加入历史
                    if not response.startswith("抱歉") and not response.startswith("错误"):
                        st.session_state["history"].append((prompt, response))
                except Exception as e:
                    error_msg = f"处理请求时出错: {e}"
                    st.error(error_msg)
                    if "connection" in str(e).lower():
                        st.session_state["api_connected"] = False
                        st.session_state["api_error"] = str(e)
                        st.error("API 连接已断开，请在侧边栏重试连接")
            
            # 重置处理标志
            st.session_state["processing"] = False
            
            # 只有在成功时才重新加载页面
            if not response.startswith("抱歉") and not response.startswith("错误"):
                st.rerun()

if __name__ == "__main__":
    main()
