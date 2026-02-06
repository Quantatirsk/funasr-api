# -*- coding: utf-8 -*-
"""
模型预加载工具
在应用启动时预加载所有需要的模型,避免首次请求时的延迟
"""

import logging

logger = logging.getLogger(__name__)


def print_model_statistics(result: dict, use_logger: bool = True):
    """
    打印模型加载统计信息

    Args:
        result: preload_models() 返回的结果字典
        use_logger: True使用logger输出（记录到日志），False使用print输出（显示到控制台）
    """
    output = logger.info if use_logger else print

    output("=" * 60)
    output("📊 模型加载统计：")
    output("-" * 60)

    loaded_models = []
    failed_models = []
    skipped_models = []
    model_index = 1  # 序号计数器

    # 统计所有ASR模型
    for model_id, status in result["asr_models"].items():
        if status["loaded"]:
            loaded_models.append(f"ASR模型({model_id})")
            output(f"   {model_index}. ✅ ASR模型({model_id}): 已加载")
            model_index += 1
        elif status["error"] is not None:
            failed_models.append(f"ASR模型({model_id})")
            if use_logger:
                logger.error(f"   {model_index}. ❌ ASR模型({model_id}): {status['error']}")
            else:
                output(f"   {model_index}. ❌ ASR模型({model_id}): {status['error']}")
            model_index += 1

    # 统计其他模型（按优化后的顺序）
    other_models = [
        ("vad_model", "语音活动检测模型(VAD)"),
        ("punc_model", "标点符号模型(离线)"),
        ("punc_realtime_model", "标点符号模型(实时)"),
        ("speaker_diarization_model", "说话人分离模型(CAM++)"),
    ]

    for key, name in other_models:
        if result[key]["loaded"]:
            loaded_models.append(name)
            output(f"   {model_index}. ✅ {name}: 已加载")
            model_index += 1
        elif result[key]["error"] is not None:
            failed_models.append(name)
            if use_logger:
                logger.error(f"   {model_index}. ❌ {name}: {result[key]['error']}")
            else:
                output(f"   {model_index}. ❌ {name}: {result[key]['error']}")
            model_index += 1
        else:
            skipped_models.append(name)
            output(f"   {model_index}. ⏭️  {name}: 已跳过")
            model_index += 1

    output("-" * 60)
    loaded_count = len(loaded_models)
    total_count = loaded_count + len(failed_models)

    if loaded_count == total_count and total_count > 0:
        output(
            f"🎉 所有模型加载完成! (成功: {loaded_count}, 跳过: {len(skipped_models)})"
        )
    elif total_count > 0:
        if use_logger:
            logger.warning(
                f"⚠️  部分模型加载失败 (成功: {loaded_count}/{total_count}, 失败: {len(failed_models)}, 跳过: {len(skipped_models)})"
            )
        else:
            output(
                f"⚠️  部分模型加载失败 (成功: {loaded_count}/{total_count}, 失败: {len(failed_models)}, 跳过: {len(skipped_models)})"
            )
    else:
        if use_logger:
            logger.warning("⚠️  没有模型被加载")
        else:
            output("⚠️  没有模型被加载")

    output("=" * 60)


def _has_cuda() -> bool:
    """检查是否有 CUDA 可用"""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _detect_qwen_model_by_vram() -> str | None:
    """根据显存检测应该使用哪个 Qwen 模型

    < 32GB 用 0.6b, >= 32GB 用 1.7b
    CPU 环境返回 None（vLLM 不支持 CPU）
    """
    if not _has_cuda():
        return None

    try:
        import torch

        min_vram = min(
            torch.cuda.get_device_properties(i).total_memory / (1024**3)
            for i in range(torch.cuda.device_count())
        )
        return "qwen3-asr-1.7b" if min_vram >= 32 else "qwen3-asr-0.6b"
    except Exception:
        return "qwen3-asr-0.6b"


def _resolve_models_to_load(all_available_models: list[str], config: str) -> list[str]:
    """解析配置，返回应加载的模型列表

    CPU 环境下自动过滤 Qwen 模型（vLLM 不支持 CPU）

    Args:
        all_available_models: 所有可用模型ID
        config: ENABLED_MODELS 配置值

    Returns:
        应加载的模型ID列表
    """
    cfg = config.strip().lower()
    has_cuda = _has_cuda()

    # all: 加载所有（CPU 下过滤 Qwen）
    if cfg == "all":
        if has_cuda:
            return all_available_models
        # CPU: 只加载非 Qwen 模型
        return [m for m in all_available_models if not m.startswith("qwen3-asr-")]

    # auto: 自动检测显存 + paraformer-large
    if cfg == "auto":
        qwen_model = _detect_qwen_model_by_vram()
        models = []
        if qwen_model and qwen_model in all_available_models:
            models.append(qwen_model)
        if "paraformer-large" in all_available_models:
            models.append("paraformer-large")
        return models

    # 其他: 精确匹配，过滤掉不存在的（CPU 下额外过滤 Qwen）
    requested = [m.strip() for m in config.split(",") if m.strip()]
    result = [m for m in requested if m in all_available_models]
    if not has_cuda:
        # CPU 环境过滤 Qwen
        result = [m for m in result if not m.startswith("qwen3-asr-")]
    return result


def preload_models() -> dict:
    """
    预加载所有需要的模型（根据 ENABLE_* 配置过滤）

    Returns:
        dict: 包含加载状态的字典
    """
    # 修复 CAM++ 配置文件（用于离线环境）
    try:
        from .download_models import fix_camplusplus_config
        fix_camplusplus_config()
    except Exception:
        pass  # 修复失败不影响启动

    result = {
        "asr_models": {},  # 所有ASR模型加载状态
        "vad_model": {"loaded": False, "error": None},
        "punc_model": {"loaded": False, "error": None},
        "punc_realtime_model": {"loaded": False, "error": None},
        "speaker_diarization_model": {"loaded": False, "error": None},
    }

    from ..core.config import settings
    from ..services.asr.registry import register_loaded_model

    # 初始化变量，避免未绑定错误
    asr_engine = None
    model_manager = None

    logger.info("=" * 60)
    logger.info("🔄 开始预加载模型...")
    logger.info(f"   配置: ENABLED_MODELS={settings.ENABLED_MODELS}")
    logger.info("=" * 60)

    # 1. 预加载所有配置的ASR模型（根据 ENABLE_* 配置过滤）
    try:
        from ..services.asr.manager import get_model_manager

        model_manager = get_model_manager()

        # 获取所有模型配置
        all_models = model_manager.list_models()
        model_ids = [m["id"] for m in all_models]

        # 根据配置解析应加载的模型
        models_to_load = _resolve_models_to_load(model_ids, settings.ENABLED_MODELS)

        # 如果没有启用任何模型，发出警告
        if not models_to_load:
            logger.warning(f"⚠️  没有启用任何 ASR 模型！请检查 ENABLED_MODELS 配置: {settings.ENABLED_MODELS}")

        logger.info(f"📋 发现 {len(model_ids)} 个模型配置，将加载 {len(models_to_load)} 个: {', '.join(models_to_load) if models_to_load else '（无）'}")

        for model_id in models_to_load:
            result["asr_models"][model_id] = {"loaded": False, "error": None}

            try:
                engine = model_manager.get_asr_engine(model_id)

                if engine.is_model_loaded():
                    result["asr_models"][model_id]["loaded"] = True
                    register_loaded_model(model_id)  # 注册到全局注册表

                    # 保存第一个成功加载的引擎引用（用于后续获取device）
                    if asr_engine is None:
                        asr_engine = engine
                else:
                    result["asr_models"][model_id]["error"] = "模型加载后未正确初始化"

                # 为 Qwen3-ASR 加载流式专用实例（完全隔离状态）
                # 仅在 ENABLE_STREAMING_VLLM=true 时加载流式实例（默认 false，节省显存）
                if settings.ENABLE_STREAMING_VLLM and model_id.startswith("qwen3-asr-"):
                    streaming_key = f"{model_id}-streaming"
                    result["asr_models"][streaming_key] = {"loaded": False, "error": None}
                    try:
                        streaming_engine = model_manager.get_asr_engine(model_id, streaming=True)
                        if streaming_engine.is_model_loaded():
                            result["asr_models"][streaming_key]["loaded"] = True
                        else:
                            result["asr_models"][streaming_key]["error"] = "模型加载后未正确初始化"
                    except Exception as e:
                        result["asr_models"][streaming_key]["error"] = str(e)

            except Exception as e:
                result["asr_models"][model_id]["error"] = str(e)

    except Exception as e:
        logger.error(f"❌ 获取模型管理器失败: {e}")
        models_to_load = []

    # 辅助函数：检查是否要加载 paraformer
    paraformer_enabled = "paraformer-large" in models_to_load

    # 2. 预加载语音活动检测模型(VAD)
    # VAD 是 Paraformer 的配套模型，只有启用 Paraformer 时才加载
    if paraformer_enabled:
        try:
            from ..services.asr.engines import get_global_vad_model

            device = asr_engine.device if asr_engine else settings.DEVICE
            vad_model = get_global_vad_model(device)

            if vad_model:
                result["vad_model"]["loaded"] = True
            else:
                result["vad_model"]["error"] = "语音活动检测模型(VAD)加载后返回None"

        except Exception as e:
            result["vad_model"]["error"] = str(e)
            logger.error(f"❌ 语音活动检测模型(VAD)加载失败: {e}")
    else:
        result["vad_model"]["error"] = "已跳过（Paraformer 未启用）"
        logger.info("⏭️  跳过 VAD 模型加载（Paraformer 未启用）")

    # 3. 预加载标点符号模型 (离线版)
    # PUNC 是 Paraformer 的配套模型，只有启用 Paraformer 时才加载
    if paraformer_enabled:
        try:
            from ..services.asr.engines import get_global_punc_model

            device = asr_engine.device if asr_engine else settings.DEVICE
            punc_model = get_global_punc_model(device)

            if punc_model:
                result["punc_model"]["loaded"] = True
            else:
                result["punc_model"]["error"] = "标点符号模型加载后返回None"

        except Exception as e:
            result["punc_model"]["error"] = str(e)
            logger.error(f"❌ 标点符号模型(离线)加载失败: {e}")
    else:
        result["punc_model"]["error"] = "已跳过（Paraformer 未启用）"
        logger.info("⏭️  跳过标点模型加载（Paraformer 未启用）")

    # 4. 预加载实时标点符号模型 (如果启用)
    if paraformer_enabled and settings.ASR_ENABLE_REALTIME_PUNC:
        try:
            from ..services.asr.engines import get_global_punc_realtime_model

            device = asr_engine.device if asr_engine else settings.DEVICE
            punc_realtime_model = get_global_punc_realtime_model(device)

            if punc_realtime_model:
                result["punc_realtime_model"]["loaded"] = True
            else:
                result["punc_realtime_model"]["error"] = "实时标点符号模型加载后返回None"

        except Exception as e:
            result["punc_realtime_model"]["error"] = str(e)
            logger.error(f"❌ 实时标点符号模型加载失败: {e}")
    else:
        skip_reason = "Paraformer 未启用" if not paraformer_enabled else "实时标点被禁用"
        result["punc_realtime_model"]["error"] = f"已跳过（{skip_reason}）"
        logger.info(f"⏭️  跳过实时标点模型加载（{skip_reason}）")

    # 5. 预加载说话人分离模型 (CAM++) - 必需模型，始终加载
    try:
        from ..utils.speaker_diarizer import get_global_diarization_pipeline

        diarization_pipeline = get_global_diarization_pipeline()

        if diarization_pipeline:
            result["speaker_diarization_model"]["loaded"] = True
        else:
            result["speaker_diarization_model"]["error"] = "说话人分离模型加载后返回None"

    except Exception as e:
        result["speaker_diarization_model"]["error"] = str(e)
        logger.error(f"❌ 说话人分离模型(CAM++)加载失败: {e}")

    return result
