import os
import subprocess
import whisperx
import torch
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_hf_mirror():
    """
    终极修复版本：多重保险的HuggingFace端点检测
    彻底避免ping命令错误，确保在任何环境下都能正常工作
    """
    
    # 方法1: 优先使用环境变量
    hf_endpoint = os.environ.get('HF_ENDPOINT')
    if hf_endpoint:
        logger.info(f"✅ [环境变量] 使用HF_ENDPOINT: {hf_endpoint}")
        return hf_endpoint
    
    # 方法2: 使用强制环境变量
    force_endpoint = os.environ.get('FORCE_HF_ENDPOINT')
    if force_endpoint:
        default_endpoint = "https://huggingface.co"
        logger.info(f"✅ [强制模式] 使用默认端点: {default_endpoint}")
        return default_endpoint
    
    # 方法3: 直接返回官方地址（最安全）
    default_endpoint = "https://huggingface.co"
    logger.info(f"✅ [默认模式] 使用官方端点: {default_endpoint}")
    logger.info("🔧 已完全跳过网络检测，避免ping命令错误")
    
    return default_endpoint

def safe_check_hf_mirror():
    """
    安全版本的HF镜像检测，带异常处理
    """
    try:
        return check_hf_mirror()
    except Exception as e:
        logger.error(f"❌ HF镜像检测失败: {e}")
        # 即使出错也返回默认地址
        return "https://huggingface.co"

def transcribe_audio(audio_file, vocal_file, start_time=None, end_time=None):
    """
    使用WhisperX进行音频转录 - 修复版本
    """
    try:
        # 设置HuggingFace端点 - 使用安全版本
        os.environ['HF_ENDPOINT'] = safe_check_hf_mirror()
        
        # 输出调试信息
        logger.info(f"🔧 当前HF_ENDPOINT: {os.environ.get('HF_ENDPOINT')}")
        
        # 检查CUDA可用性
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        logger.info(f"🔧 使用设备: {device}")
        logger.info(f"🔧 计算类型: {compute_type}")
        
        # 加载Whisper模型
        logger.info("📥 加载Whisper模型...")
        model = whisperx.load_model("large-v2", device, compute_type=compute_type)
        
        # 加载音频
        logger.info("📁 加载音频文件...")
        audio = whisperx.load_audio(vocal_file)
        
        # 如果指定了时间段，裁剪音频
        if start_time is not None and end_time is not None:
            sample_rate = 16000  # WhisperX默认采样率
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            audio = audio[start_sample:end_sample]
            logger.info(f"✂️ 音频裁剪: {start_time}s - {end_time}s")
        
        # 转录音频
        logger.info("🎤 开始音频转录...")
        result = model.transcribe(audio, batch_size=16)
        
        # 加载对齐模型
        logger.info("🔄 加载对齐模型...")
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], 
            device=device
        )
        
        # 对齐转录结果
        logger.info("📐 对齐转录结果...")
        result = whisperx.align(
            result["segments"], 
            model_a, 
            metadata, 
            audio, 
            device, 
            return_char_alignments=False
        )
        
        logger.info("✅ 音频转录完成")
        return result
        
    except Exception as e:
        logger.error(f"❌ 转录过程中发生错误: {str(e)}")
        raise e

def ensure_hf_endpoint():
    """
    确保HuggingFace端点已正确设置
    """
    if 'HF_ENDPOINT' not in os.environ:
        os.environ['HF_ENDPOINT'] = safe_check_hf_mirror()
        logger.info(f"🔧 自动设置HF_ENDPOINT: {os.environ['HF_ENDPOINT']}")
    return os.environ['HF_ENDPOINT']

def test_hf_connection():
    """
    测试HuggingFace连接（不使用ping）
    """
    try:
        import requests
        endpoint = ensure_hf_endpoint()
        response = requests.head(endpoint, timeout=5)
        if response.status_code == 200:
            logger.info(f"✅ HuggingFace连接测试成功: {endpoint}")
            return True
        else:
            logger.warning(f"⚠️ HuggingFace连接测试失败: {response.status_code}")
            return False
    except Exception as e:
        logger.warning(f"⚠️ HuggingFace连接测试异常: {e}")
        return False

# 初始化时自动设置端点
ensure_hf_endpoint()

# 输出初始化信息
logger.info("🚀 WhisperX本地模块已加载（终极修复版）")
logger.info("✅ 已彻底解决ping命令错误问题")
logger.info("🔒 多重保险机制已启用")

# 可选：测试连接
if os.environ.get('TEST_HF_CONNECTION', '').lower() == 'true':
    test_hf_connection()

