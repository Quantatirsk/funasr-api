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


def preload_models() -> dict:
    """
    预加载所有需要的模型（所有配置的ASR模型）

    Returns:
        dict: 包含加载状态的字典
    """
    result = {
        "asr_models": {},  # 所有ASR模型加载状态
        "vad_model": {"loaded": False, "error": None},
        "punc_model": {"loaded": False, "error": None},
        "punc_realtime_model": {"loaded": False, "error": None},
        "speaker_diarization_model": {"loaded": False, "error": None},
    }

    from ..core.config import settings

    # 初始化变量，避免未绑定错误
    asr_engine = None
    model_manager = None

    logger.info("=" * 60)
    logger.info("🔄 开始预加载所有模型...")
    logger.info("=" * 60)

    # 1. 预加载所有配置的ASR模型
    try:
        from ..services.asr.manager import get_model_manager

        model_manager = get_model_manager()

        # 获取所有模型配置
        all_models = model_manager.list_models()
        model_ids = [m["id"] for m in all_models]

        # 根据配置过滤要加载的模型
        # 只加载默认模型和显式指定的模型
        default_model = model_manager._default_model_id
        models_to_load = [default_model] if default_model in model_ids else []

        # 如果默认模型是 qwen3-asr-0.6b，跳过 qwen3-asr-1.7b
        # 如果默认模型是 qwen3-asr-1.7b，加载它（以及可能的 0.6b 用于其他用途）

        # 添加 paraformer-large（如果配置了）
        if "paraformer-large" in model_ids and settings.ASR_MODEL_MODE in ["all", "offline"]:
            models_to_load.append("paraformer-large")

        logger.info(f"📋 发现 {len(model_ids)} 个模型配置，将加载 {len(models_to_load)} 个: {', '.join(models_to_load)}")

        for model_id in models_to_load:
            result["asr_models"][model_id] = {"loaded": False, "error": None}

            try:
                logger.info(f"📥 正在加载ASR模型: {model_id}...")
                engine = model_manager.get_asr_engine(model_id)

                if engine.is_model_loaded():
                    result["asr_models"][model_id]["loaded"] = True
                    logger.info(f"✅ ASR模型加载成功: {model_id}")

                    # 保存第一个成功加载的引擎引用（用于后续获取device）
                    if asr_engine is None:
                        asr_engine = engine
                else:
                    result["asr_models"][model_id]["error"] = "模型加载后未正确初始化"
                    logger.warning(f"⚠️  ASR模型 {model_id} 加载后未正确初始化")

                # 为 Qwen3-ASR 加载流式专用实例（完全隔离状态）
                # 根据实际加载的模型决定流式实例
                if model_id.startswith("qwen3-asr-") and model_id in ["qwen3-asr-1.7b", "qwen3-asr-0.6b"]:
                    streaming_key = f"{model_id}-streaming"
                    result["asr_models"][streaming_key] = {"loaded": False, "error": None}
                    try:
                        logger.info(f"📥 正在加载ASR模型流式实例: {streaming_key}...")
                        streaming_engine = model_manager.get_asr_engine(model_id, streaming=True)
                        if streaming_engine.is_model_loaded():
                            result["asr_models"][streaming_key]["loaded"] = True
                            logger.info(f"✅ ASR模型流式实例加载成功: {streaming_key}")
                        else:
                            result["asr_models"][streaming_key]["error"] = "模型加载后未正确初始化"
                            logger.warning(f"⚠️  ASR模型流式实例 {streaming_key} 加载后未正确初始化")
                    except Exception as e:
                        result["asr_models"][streaming_key]["error"] = str(e)
                        logger.error(f"❌ ASR模型流式实例 {streaming_key} 加载失败: {e}")

            except Exception as e:
                result["asr_models"][model_id]["error"] = str(e)
                logger.error(f"❌ ASR模型 {model_id} 加载失败: {e}")

    except Exception as e:
        logger.error(f"❌ 获取模型管理器失败: {e}")

    # 3. 预加载语音活动检测模型(VAD) (如果ASR模式包含离线模型)
    if settings.ASR_MODEL_MODE.lower() in ["all", "offline"]:
        try:
            logger.info("📥 正在加载语音活动检测模型(VAD)...")
            from ..services.asr.engines import get_global_vad_model

            device = asr_engine.device if asr_engine else settings.DEVICE
            vad_model = get_global_vad_model(device)

            if vad_model:
                result["vad_model"]["loaded"] = True
                logger.info("✅ 语音活动检测模型(VAD)加载成功")
            else:
                result["vad_model"]["error"] = "语音活动检测模型(VAD)加载后返回None"
                logger.warning("⚠️  语音活动检测模型(VAD)加载后返回None")

        except Exception as e:
            result["vad_model"]["error"] = str(e)
            logger.error(f"❌ 语音活动检测模型(VAD)加载失败: {e}")
    else:
        logger.info("⏭️  跳过语音活动检测模型(VAD)加载 (ASR_MODEL_MODE=realtime)")

    # 4. 预加载标点符号模型 (离线版)
    try:
        logger.info("📥 正在加载标点符号模型(离线)...")
        from ..services.asr.engines import get_global_punc_model

        device = asr_engine.device if asr_engine else settings.DEVICE
        punc_model = get_global_punc_model(device)

        if punc_model:
            result["punc_model"]["loaded"] = True
            logger.info("✅ 标点符号模型(离线)加载成功")
        else:
            result["punc_model"]["error"] = "标点符号模型加载后返回None"
            logger.warning("⚠️  标点符号模型(离线)加载后返回None")

    except Exception as e:
        result["punc_model"]["error"] = str(e)
        logger.error(f"❌ 标点符号模型(离线)加载失败: {e}")

    # 5. 预加载实时标点符号模型 (如果启用)
    if settings.ASR_ENABLE_REALTIME_PUNC:
        try:
            logger.info("📥 正在加载实时标点符号模型...")
            from ..services.asr.engines import get_global_punc_realtime_model

            device = asr_engine.device if asr_engine else settings.DEVICE
            punc_realtime_model = get_global_punc_realtime_model(device)

            if punc_realtime_model:
                result["punc_realtime_model"]["loaded"] = True
                logger.info("✅ 实时标点符号模型加载成功")
            else:
                result["punc_realtime_model"]["error"] = "实时标点符号模型加载后返回None"
                logger.warning("⚠️  实时标点符号模型加载后返回None")

        except Exception as e:
            result["punc_realtime_model"]["error"] = str(e)
            logger.error(f"❌ 实时标点符号模型加载失败: {e}")
    else:
        logger.info("⏭️  跳过实时标点符号模型加载 (ASR_ENABLE_REALTIME_PUNC=False)")

    # 6. 预加载说话人分离模型 (CAM++)
    try:
        logger.info("📥 正在加载说话人分离模型(CAM++)...")
        from ..utils.speaker_diarizer import get_global_diarization_pipeline

        diarization_pipeline = get_global_diarization_pipeline()

        if diarization_pipeline:
            result["speaker_diarization_model"]["loaded"] = True
            logger.info("✅ 说话人分离模型(CAM++)加载成功")
        else:
            result["speaker_diarization_model"]["error"] = "说话人分离模型加载后返回None"
            logger.warning("⚠️  说话人分离模型(CAM++)加载后返回None")

    except Exception as e:
        result["speaker_diarization_model"]["error"] = str(e)
        logger.error(f"❌ 说话人分离模型(CAM++)加载失败: {e}")

    # 打印统计结果到日志
    print_model_statistics(result, use_logger=True)

    return result
