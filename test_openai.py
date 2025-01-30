from openai import OpenAI
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def test_openai_connection():
    try:
        # 初始化 OpenAI 客户端
        client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_BASE_URL', "https://api.openai.com/v1")
        )

        # 获取可用模型列表
        print("正在获取可用模型列表...")
        models = client.models.list()
        print("\n可用的模型：")
        for model in models.data:
            print(f"- {model.id}")

        # 测试对话
        print("\n测试对话...")
        response = client.chat.completions.create(
            model="gpt-4",  # 先用标准模型测试
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! What models are you?"}
            ]
        )
        print("\n模型响应：")
        print(response.choices[0].message.content)

    except Exception as e:
        print(f"\n发生错误：{str(e)}")

if __name__ == "__main__":
    print("OpenAI API 测试")
    print("-" * 50)
    print(f"API Key: {os.getenv('OPENAI_API_KEY')[:10]}...")
    print(f"Base URL: {os.getenv('OPENAI_BASE_URL')}")
    print("-" * 50)
    test_openai_connection() 