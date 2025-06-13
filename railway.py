import os

# 检查Claude API配置
def check_claude_config():
    api_key = os.environ.get('sk-QzOKB3kU82rZcuh6Sy13oGm8pqjEFLQg5pQ3WlBwFUoDv08E')
    base_url = os.environ.get('BASE_URL', 'https://api.302.ai')
    model = os.environ.get('MODEL', 'claude-3-5-sonnet-20241022')
    
    if not api_key:
        logger.error("❌ API_KEY环境变量未设置")
        return False
        
    if not api_key.startswith('sk-302ai-'):
        logger.warning("⚠️ API密钥格式可能不正确")
    
    logger.info(f"✅ Claude配置检查通过")
    logger.info(f"📡 Base URL: {base_url}")
    logger.info(f"🤖 Model: {model}")
    
    return True
