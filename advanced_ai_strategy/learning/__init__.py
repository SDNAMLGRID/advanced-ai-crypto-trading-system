"""
Advanced AI Strategy Learning & Adaptation Module
=================================================

This module provides sophisticated learning and adaptation capabilities for continuous
strategy improvement and market adaptation:

- Continuous Learning: Online learning algorithms for real-time strategy adaptation
- Performance Monitoring: Real-time performance tracking with drift detection
- Adaptive Management: Dynamic strategy parameter adjustment

Key Features:
- Online learning with concept drift detection
- Real-time performance monitoring and alerting
- Dynamic parameter adaptation based on market conditions
- Regime-aware learning and adaptation
"""

# Continuous Learning Engine
from .continuous_learner import (
    ContinuousLearner,
    LearningConfig,
    LearningState,
    AdaptationResult
)

# Performance Monitoring
from .performance_monitor import (
    PerformanceMonitor,
    MonitorConfig,
    PerformanceAlert,
    DriftDetection
)

# Adaptive Strategy Management
from .adaptive_manager import (
    AdaptiveManager,
    AdaptationStrategy,
    ParameterUpdate,
    AdaptationEvent
)

# TODO: Future components to implement
# Model Update Framework
# from .model_updater import (
#     ModelUpdater,
#     UpdateConfig,
#     UpdateResult,
#     ModelVersion
# )

# Ensemble Learning
# from .ensemble_learner import (
#     EnsembleLearner,
#     EnsembleConfig,
#     EnsembleModel,
#     WeightingScheme
# )

# Real-time Regime Detection
# from .regime_detector import (
#     RealtimeRegimeDetector,
#     RegimeChange,
#     RegimeConfig,
#     RegimeSignal
# )

# Meta-Learning Framework
# from .meta_learner import (
#     MetaLearner,
#     MetaConfig,
#     MetaKnowledge,
#     LearningStrategy
# )

__all__ = [
    # Continuous Learning
    'ContinuousLearner',
    'LearningConfig',
    'LearningState',
    'AdaptationResult',
    
    # Performance Monitoring
    'PerformanceMonitor',
    'MonitorConfig',
    'PerformanceAlert',
    'DriftDetection',
    
    # Adaptive Management
    'AdaptiveManager',
    'AdaptationStrategy',
    'ParameterUpdate',
    'AdaptationEvent',
    
    # TODO: Future components
    # 'ModelUpdater',
    # 'UpdateConfig',
    # 'UpdateResult',
    # 'ModelVersion',
    # 'EnsembleLearner',
    # 'EnsembleConfig',
    # 'EnsembleModel',
    # 'WeightingScheme',
    # 'RealtimeRegimeDetector',
    # 'RegimeChange',
    # 'RegimeConfig',
    # 'RegimeSignal',
    # 'MetaLearner',
    # 'MetaConfig',
    # 'MetaKnowledge',
    # 'LearningStrategy'
] 