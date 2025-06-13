import os
import subprocess
import sys

# Railway环境配置
PORT = os.environ.get('PORT', '8080')

print("🚄 Railway VideoLingo 启动中...")

# 检查环境变量
required_env = ['API_KEY', 'BASE_URL', 'MODEL']
for env_var in required_env:
    if not os.environ.get(env_var):
        print(f"⚠️ 警告：环境变量 {env_var} 未设置")

# 创建必要目录
os.makedirs('output/log', exist_ok=True)
os.makedirs('_model_cache', exist_ok=True)

print("✅ 环境检查完成")
print(f"🚀 在端口 {PORT} 启动服务...")

# 启动Streamlit
try:
    subprocess.run([
        'streamlit', 'run', 'st.py',
        '--server.port', PORT,
        '--server.address', '0.0.0.0',
        '--server.headless', 'true',
        '--server.enableCORS', 'false'
    ])
except Exception as e:
    print(f"❌ 启动失败：{e}")
    sys.exit(1)
