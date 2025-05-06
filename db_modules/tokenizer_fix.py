"""
修复tokenizer相关问题的模块
"""
import logging
import random
import string
from typing import Optional, List, Union

logger = logging.getLogger(__name__)

def find_safe_rare_token(tokenizer, retries: int = 5) -> str:
    """
    安全地找到一个罕见的token，避免NoneType错误
    
    Args:
        tokenizer: 要使用的tokenizer
        retries: 尝试次数
    
    Returns:
        一个可用的罕见token字符串
    """
    logger.info("尝试找到安全的罕见token...")
    
    # 预定义的一些已知的安全token
    safe_tokens = ["sks", "xzs", "zxq", "vxz"]
    
    # 先尝试使用预定义的token
    for token in safe_tokens:
        try:
            token_ids = tokenizer.encode(token, add_special_tokens=False)
            decoded = tokenizer.decode(token_ids)
            if decoded and isinstance(decoded, str) and len(decoded.strip()) > 0:
                logger.info(f"找到安全token: {token}")
                return token
        except Exception as e:
            logger.warning(f"尝试token '{token}'时出错: {e}")
    
    # 如果预定义token不起作用，尝试生成随机token
    for _ in range(retries):
        try:
            # 生成一个随机字符串
            rnd_string = ''.join(random.choices(string.ascii_lowercase, k=3))
            token_ids = tokenizer.encode(rnd_string, add_special_tokens=False)
            
            # 安全解码
            if not token_ids:
                continue
                
            decoded = safe_decode(tokenizer, token_ids)
            if decoded and len(decoded.strip()) > 0:
                logger.info(f"生成的随机安全token: {rnd_string}")
                return rnd_string
        except Exception as e:
            logger.warning(f"尝试随机token生成时出错: {e}")
    
    # 如果所有尝试都失败，返回安全的后备token
    logger.warning("无法生成安全token，使用后备token")
    return "person"

def safe_decode(tokenizer, token_ids: List[int]) -> str:
    """
    安全地解码token ID，避免NoneType错误
    
    Args:
        tokenizer: 要使用的tokenizer
        token_ids: 要解码的token ID列表
    
    Returns:
        解码后的字符串
    """
    try:
        # 尝试解码前过滤空值
        valid_ids = [tid for tid in token_ids if tid is not None]
        if not valid_ids:
            return ""
            
        # 使用原始解码方法
        return tokenizer.decode(valid_ids).strip()
    except Exception as e:
        logger.error(f"Token解码错误: {e}")
        return ""

def patch_tokenizer(tokenizer):
    """
    修补tokenizer以避免常见错误
    
    Args:
        tokenizer: 要修补的tokenizer
        
    Returns:
        修补后的tokenizer
    """
    # 存储原始方法
    original_decode = tokenizer.decode
    original_convert = getattr(tokenizer, 'convert_tokens_to_string', None)
    
    # 添加安全wrapper
    def safe_decode_wrapper(token_ids, *args, **kwargs):
        try:
            # 过滤None值
            valid_ids = [tid for tid in token_ids if tid is not None]
            if not valid_ids:
                return ""
            return original_decode(valid_ids, *args, **kwargs)
        except Exception as e:
            logger.warning(f"解码错误，使用安全后备: {e}")
            return "[DECODE_ERROR]"
    
    if original_convert:
        def safe_convert_wrapper(tokens, *args, **kwargs):
            try:
                # 过滤None值
                valid_tokens = [t for t in tokens if t is not None]
                if not valid_tokens:
                    return ""
                return original_convert(valid_tokens, *args, **kwargs)
            except Exception as e:
                logger.warning(f"转换令牌错误，使用安全后备: {e}")
                return "[TOKEN_ERROR]"
                
        tokenizer.convert_tokens_to_string = safe_convert_wrapper
    
    tokenizer.decode = safe_decode_wrapper
    return tokenizer
