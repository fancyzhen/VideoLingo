import os
import subprocess
import whisperx
import torch
from pathlib import Path

def check_hf_mirror():
    """
    紧急修复版本：直接返回HuggingFace官方地址
    解决Railway环境中ping命令不存在的问题
    """
    # 直接返回官方地址，跳过所有网络检测
    hf_endpoint = "https://huggingface.co"
    print(f"✅ 使用HuggingFace官方端点: {hf_endpoint}")
    print("🔧 已跳过ping检测，避免容器环境错误")
    return hf_endpoint

def transcribe_audio(audio_file, vocal_file, start_time=None, end_time=None):
    """
    使用WhisperX进行音频转录
    """
    try:
        # 设置HuggingFace端点
        os.environ['HF_ENDPOINT'] = check_hf_mirror()
        
        # 检查CUDA可用性
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        print(f"🔧 使用设备: {device}")
        print(f"🔧 计算类型: {compute_type}")
        
        # 加载Whisper模型
        model = whisperx.load_model("large-v2", device, compute_type=compute_type)
        
        # 加载音频
        audio = whisperx.load_audio(vocal_file)
        
        # 如果指定了时间段，裁剪音频
        if start_time is not None and end_time is not None:
            sample_rate = 16000  # WhisperX默认采样率
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            audio = audio[start_sample:end_sample]
        
        # 转录音频
        print("🎤 开始音频转录...")
        result = model.transcribe(audio, batch_size=16)
        
        # 加载对齐模型
        print("🔄 加载对齐模型...")
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], 
            device=device
        )
        
        # 对齐转录结果
        print("📐 对齐转录结果...")
        result = whisperx.align(
            result["segments"], 
            model_a, 
            metadata, 
            audio, 
            device, 
            return_char_alignments=False
        )
        
        print("✅ 音频转录完成")
        return result
        
    except Exception as e:
        print(f"❌ 转录过程中发生错误: {str(e)}")
        raise e

def get_whisper_result():
    """
    获取Whisper转录结果的主函数
    """
    try:
        # 这里应该根据实际的音频文件路径进行调用
        # 示例调用（实际使用时需要传入正确的文件路径）
        # result = transcribe_audio("input.wav", "vocal.wav")
        # return result
        pass
    except Exception as e:
        print(f"❌ 获取转录结果失败: {str(e)}")
        raise e

# 兼容性函数
def ensure_hf_endpoint():
    """
    确保HuggingFace端点已设置
    """
    if 'HF_ENDPOINT' not in os.environ:
        os.environ['HF_ENDPOINT'] = check_hf_mirror()
    return os.environ['HF_ENDPOINT']

# 初始化时自动设置端点
ensure_hf_endpoint()

print("🚀 WhisperX本地模块已加载（修复版）")
print("✅ 已解决ping命令错误问题")

