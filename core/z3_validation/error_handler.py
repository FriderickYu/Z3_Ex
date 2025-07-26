import logging

logger = logging.getLogger(__name__)

class Z3ErrorHandler:
    """用于处理异常与调试日志"""

    @staticmethod
    def handle_exception(e: Exception, context: str = "") -> dict:
        logger.error(f"[Z3Error] Exception in {context}: {e}", exc_info=True)
        return {
            "status": "ERROR",
            "error": str(e),
            "context": context
        }