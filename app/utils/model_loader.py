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

    # 统计默认ASR模型
    if result["asr_default_model"]["loaded"]:
        model_id = result["asr_default_model"]["model_id"]
        loaded_models.append(f"默认ASR模型({model_id})")
        output(f"   {model_index}. ✅ 默认ASR模型({model_id}): 已加载")
        model_index += 1
    elif result["asr_default_model"]["error"] is not None:
        failed_models.append("默认ASR模型")
        if use_logger:
            logger.error(f"   {model_index}. ❌ 默认ASR模型: {result['asr_default_model']['error']}")
        else:
            output(f"   {model_index}. ❌ 默认ASR模型: {result['asr_default_model']['error']}")
        model_index += 1

    # 统计自定义ASR模型
    for model_id, status in result["asr_custom_models"].items():
        if status["loaded"]:
            loaded_models.append(f"自定义ASR模型({model_id})")
            output(f"   {model_index}. ✅ 自定义ASR模型({model_id}): 已加载")
            model_index += 1
        elif status["error"] is not None:
            failed_models.append(f"自定义ASR模型({model_id})")
            if use_logger:
                logger.error(f"   {model_index}. ❌ 自定义ASR模型({model_id}): {status['error']}")
            else:
                output(f"   {model_index}. ❌ 自定义ASR模型({model_id}): {status['error']}")
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
    预加载所有需要的模型

    Returns:
        dict: 包含加载状态的字典
    """
    result = {
        "asr_default_model": {"loaded": False, "error": None, "model_id": None},
        "asr_custom_models": {},  # 动态添加自定义ASR模型
        "vad_model": {"loaded": False, "error": None},
        "punc_model": {"loaded": False, "error": None},
        "punc_realtime_model": {"loaded": False, "error": None},
        "speaker_diarization_model": {"loaded": False, "error": None},
    }

    from ..core.config import settings

    # 初始化变量，避免未绑定错误
    asr_engine = None

    logger.info("=" * 60)
    logger.info("🔄 开始预加载模型...")
    logger.info("=" * 60)

    # 1. 预加载默认ASR模型
    try:
        logger.info("📥 正在加载默认ASR模型...")
        from ..services.asr.manager import get_model_manager

        model_manager = get_model_manager()
        asr_engine = model_manager.get_asr_engine()  # 加载默认模型

        if asr_engine.is_model_loaded():
            default_model_id = model_manager._default_model_id
            result["asr_default_model"]["loaded"] = True
            result["asr_default_model"]["model_id"] = default_model_id
            logger.info(f"✅ 默认ASR模型加载成功: {default_model_id}")

            # 根据ASR_MODEL_MODE显示加载的模型类型
            mode = settings.ASR_MODEL_MODE.lower()
            if mode == "all":
                logger.info(
                    f"   - 离线模型: {'✓' if getattr(asr_engine, 'offline_model', None) else '✗'}"
                )
                logger.info(
                    f"   - 实时模型: {'✓' if getattr(asr_engine, 'realtime_model', None) else '✗'}"
                )
            elif mode == "offline":
                logger.info("   - 离线模型: ✓")
            elif mode == "realtime":
                logger.info("   - 实时模型: ✓")
        else:
            result["asr_default_model"]["error"] = "ASR模型加载后未正确初始化"
            logger.warning("⚠️  默认ASR模型加载后未正确初始化")

    except Exception as e:
        result["asr_default_model"]["error"] = str(e)
        logger.error(f"❌ 默认ASR模型加载失败: {e}")

    # 2. 预加载自定义ASR模型（如果配置了AUTO_LOAD_CUSTOM_ASR_MODELS）
    if settings.AUTO_LOAD_CUSTOM_ASR_MODELS:
        custom_model_ids = [
            m.strip()
            for m in settings.AUTO_LOAD_CUSTOM_ASR_MODELS.split(",")
            if m.strip()
        ]

        logger.info(
            f"📥 配置了自定义ASR模型加载: {', '.join(custom_model_ids)}"
        )

        for model_id in custom_model_ids:
            result["asr_custom_models"][model_id] = {"loaded": False, "error": None}

            try:
                logger.info(f"📥 正在加载自定义ASR模型: {model_id}...")
                from ..services.asr.manager import get_model_manager

                model_manager = get_model_manager()

                # 检查模型是否在配置中
                try:
                    model_config = model_manager.get_model_config(model_id)
                except Exception as config_error:
                    result["asr_custom_models"][model_id]["error"] = (
                        f"模型配置不存在: {config_error}"
                    )
                    logger.error(f"❌ 自定义ASR模型 {model_id} 配置不存在: {config_error}")
                    continue

                # 加载模型
                custom_engine = model_manager.get_asr_engine(model_id)

                if custom_engine.is_model_loaded():
                    result["asr_custom_models"][model_id]["loaded"] = True
                    logger.info(f"✅ 自定义ASR模型加载成功: {model_id}")
                    logger.info(f"   - 引擎: {model_config.engine}")
                    logger.info(f"   - 支持实时: {model_config.supports_realtime}")
                else:
                    result["asr_custom_models"][model_id]["error"] = "模型加载后未正确初始化"
                    logger.warning(f"⚠️  自定义ASR模型 {model_id} 加载后未正确初始化")

            except Exception as e:
                result["asr_custom_models"][model_id]["error"] = str(e)
                logger.error(f"❌ 自定义ASR模型 {model_id} 加载失败: {e}")
    else:
        logger.info("⏭️  未配置自定义ASR模型加载 (AUTO_LOAD_CUSTOM_ASR_MODELS为空)")

    # 3. 预加载语音活动检测模型(VAD) (如果ASR模式包含离线模型)
    if settings.ASR_MODEL_MODE.lower() in ["all", "offline"]:
        try:
            logger.info("📥 正在加载语音活动检测模型(VAD)...")
            from ..services.asr.engine import get_global_vad_model

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
        from ..services.asr.engine import get_global_punc_model

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
            from ..services.asr.engine import get_global_punc_realtime_model

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
