"""
DreamBooth训练模块化组件
包含训练、内存优化和调试工具
"""
# 确保模块可以被正确导入

try:
    from .memory_optimization import (
        get_optimal_settings, optimize_for_inference, 
        optimize_model_for_training, track_memory_usage,
        aggressive_memory_cleanup, print_memory_stats
    )
except ImportError:
    pass

try:
    from .debugging import (
        analyze_training_failure, create_debug_report, 
        get_gpu_info, print_memory_optimization_suggestions
    )
except ImportError:
    pass
