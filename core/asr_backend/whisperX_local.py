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
    Replit优化版本：直接返回HuggingFace官方地址
    专门为Replit环境优化，避免任何网络检测问题
    """
    
    # 直接返回官方地址，Replit环境下最稳定
    hf_endpoint = "https://huggingface.co"
    
    logger.info(f"🔧 [Replit版本] 使用HuggingFace端点: {hf_endpoint}")
    logger.info("✅ [Replit版本] 跳过网络检测，确保稳定运行")
    
    return hf_endpoint

def transcribe_audio(audio_file, vocal_file, start_time=None, end_time=None):
    """
    使用WhisperX进行音频转录 - Replit优化版本
    """
    try:
        # 设置HuggingFace端点
        os.environ['HF_ENDPOINT'] = check_hf_mirror()
        
        # 输出调试信息
        logger.info(f"🔧 当前HF_ENDPOINT: {os.environ.get('HF_ENDPOINT')}")
        logger.info("🚀 [Replit] 开始WhisperX音频转录...")
        
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
        
        logger.info("✅ [Replit] 音频转录完成")
        return result
        
    except Exception as e:
        logger.error(f"❌ [Replit] 转录过程中发生错误: {str(e)}")
        raise e

def ensure_hf_endpoint():
    """
    确保HuggingFace端点已正确设置 - Replit版本
    """
    if 'HF_ENDPOINT' not in os.environ:
        os.environ['HF_ENDPOINT'] = check_hf_mirror()
        logger.info(f"🔧 [Replit] 自动设置HF_ENDPOINT: {os.environ['HF_ENDPOINT']}")
    return os.environ['HF_ENDPOINT']

# 初始化时自动设置端点
ensure_hf_endpoint()

# 输出初始化信息
logger.info("🚀 WhisperX本地模块已加载（Replit优化版）")
logger.info("✅ 专为Replit环境优化，无ping命令依赖")
logger.info("🔒 稳定性和兼容性已最大化")

