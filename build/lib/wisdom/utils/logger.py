# wisdom/utils/logger.py
import logging
from logging import handlers
import time

# Logger configuration
def configure_logging(enable_logging: bool, args, level: str = "info") -> logging.Logger:
    if not enable_logging:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        return logging.getLogger(__name__)

    log_level = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "crit": logging.CRITICAL,
    }.get(level.lower(), logging.INFO)
    
    start_ms = int(time.time() * 1000)
    timestamp = time.strftime("%Y%m%d‑%H%M%S", time.localtime(start_ms / 1000))
    logfile = f"{args.log_path}-{args.model}-{args.dataset}-{timestamp}.log"

    logger = logging.getLogger("Wisdom")
    logger.setLevel(log_level)
    
    handler = logging.FileHandler(logfile)
    formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.debug(
        "[=== Model: %s, Dataset: %s, Layers_Index: %s, Topk: %s ===]",
        args.model,
        args.dataset,
        args.layer_index,
        args.top_m_neurons,
    )
    return logger