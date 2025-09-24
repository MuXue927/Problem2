# ALNS算法配置文件

class ALNSConfig:
    """ALNS算法配置类"""
    # 数据集类型与编号参数（集中管理）
    DATASET_TYPE = 'large'   # 使用的数据集类型, 可选: 'small', 'medium', 'large'
    DATASET_IDX = 2          # 使用的算例编号, 可选: 1~30

    # Destroy算子参数
    RANDOM_REMOVAL_DEGREE = 0.25  # 随机移除比例
    SHAW_REMOVAL_DEGREE = 0.3  # Shaw移除比例
    
    # periodic shaw removal参数, 参数较多以字典形式存储
    PERIODIC_SHAW_PARAMS = {
        'degree': 0.3,
        'alpha': 0.4,
        'beta': 0.3,
        'gamma': 0.3,
        'k_clusters': 3
    }
    
    # Repair算子参数
    
    # learning_based_repair参数, 以字典形式存储
    LEARNING_BASED_REPAIR_PARAMS = {
        'model_type': 'adaptive',     # 'linear'(Ridge), 'random_forest'(RandomForest), 'adaptive'(自动选择两种之一)
        'min_score': 0.4,             # 最小得分阈值
        'initial_sample_size': 25,    # 初始训练样本量阈值, 控制进入ML训练的时机, 宜取适中值
        'adaptive_sample_size': 200,  # 自适应训练样本量阈值, 不宜过大
        'retrain_interval': 80        # 重新训练的迭代间隔
    }

    # 算法参数
    # 停止准则配置方案:
    # 使用组合停止准则：
            # 小规模数据集: 300次迭代 OR 900秒运行时间
            # 中规模数据集: 600次迭代 OR 1800秒运行时间
            # 大规模数据集: 1000次迭代 OR 3600秒运行时间
    SEED = 15926535    # 随机种子
    MAX_RUNTIME = 3600  # 最大运行时间（秒） 
    MAX_ITERATIONS_NO_IMPROVEMENT = 100  # 最大无改进迭代次数, 未启用
    MAX_ITERATIONS = 1000  # 最大迭代次数

    # 模拟退火参数
    SA_START_TEMP = 1000
    SA_END_TEMP = 1
    SA_STEP = 1 - 1e-3

    # 轮盘赌选择参数
    ROULETTE_SCORES = [5, 2, 1, 0.5]
    ROULETTE_DECAY = 0.8
    ROULETTE_SEG_LENGTH = 500

    # 多个random removal算子配置
    RANDOM_REMOVAL_VARIANTS = [
        {'degree': 0.15, 'name': 'random_gentle'},
        {'degree': 0.25, 'name': 'random_normal'},
        {'degree': 0.35, 'name': 'random_aggressive'},
    ]

    @classmethod
    def get_destroy_params(cls):
        """获取destroy算子参数"""
        return {
            'random_removal_degree': cls.RANDOM_REMOVAL_DEGREE,
            'shaw_removal_degree': cls.SHAW_REMOVAL_DEGREE,
            'periodic_shaw_params': cls.PERIODIC_SHAW_PARAMS
        }

    @classmethod
    def get_random_removal_variants(cls):
        """获取random removal的多个变体配置"""
        return cls.RANDOM_REMOVAL_VARIANTS
