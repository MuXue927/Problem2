import os
import json
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OutPutData:
    input_file_loc: str
    dataset_name: str
    
    # 数据属性，使用post_init=True延迟初始化
    order_fulfill: Dict[Tuple[int, str, str, str], int] = field(init=False)
    vehicle_load: Dict[Tuple[int, str, str, int, str], float] = field(init=False)
    non_fulfill: Dict[Tuple[str, str], int] = field(init=False)
    sku_inv_left: Dict[Tuple[str, str, int], int] = field(init=False)
    
    # DataFrame缓存
    df_result: Optional[pd.DataFrame] = field(init=False, default=None)
    df_details: Optional[pd.DataFrame] = field(init=False, default=None)
    df_summary: Optional[pd.DataFrame] = field(init=False, default=None)
    df_non_fulfill: Optional[pd.DataFrame] = field(init=False, default=None)
    df_sku_inv_left: Optional[pd.DataFrame] = field(init=False, default=None)
    
    def __post_init__(self):
        """初始化数据结构"""
        self.order_fulfill = {}
        self.vehicle_load = {}
        self.non_fulfill = {}
        self.sku_inv_left = {}
    
    def _reorganize_dataframe(self, df: pd.DataFrame, title_map: Dict[str, str]) -> pd.DataFrame:
        """
        重新组织DataFrame的列名和顺序
        
        Parameters:
        -----------
        df : pd.DataFrame
            输入的DataFrame
        title_map : Dict[str, str]
            列名映射字典
            
        Returns:
        --------
        pd.DataFrame : 重新组织后的DataFrame
        """
        # 创建副本避免修改原始数据
        df_copy = df.copy()
        
        # 重命名列
        df_copy.rename(columns=title_map, inplace=True)
        
        # 重新排序列（只包含存在的列）
        ordered_cols = [col for col in title_map.values() if col in df_copy.columns]
        df_copy = df_copy.reindex(columns=ordered_cols)
        
        return df_copy
    
    def _load_and_process_csv(self, file_path: str, title_map: Dict[str, str], 
                             dtype_map: Dict[str, type]) -> pd.DataFrame:
        """
        加载并处理CSV文件的通用方法
        
        Parameters:
        -----------
        file_path : str
            文件路径
        title_map : Dict[str, str]
            列名映射
        dtype_map : Dict[str, type]
            数据类型映射
            
        Returns:
        --------
        pd.DataFrame : 处理后的DataFrame
        """
        if not os.path.exists(file_path):
            logger.warning(f"文件不存在: {file_path}")
            # 返回空DataFrame但包含正确的列名
            empty_df = pd.DataFrame(columns=list(title_map.values()))
            return self._apply_dtypes(empty_df, dtype_map)
        
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path, header=0)
            logger.info(f"成功加载文件: {file_path}, 形状: {df.shape}")
            
            # 重新组织列名
            df = self._reorganize_dataframe(df, title_map)
            
            # 应用数据类型
            df = self._apply_dtypes(df, dtype_map)
            
            return df
            
        except Exception as e:
            logger.error(f"加载文件失败 {file_path}: {str(e)}")
            # 返回空DataFrame
            empty_df = pd.DataFrame(columns=list(title_map.values()))
            return self._apply_dtypes(empty_df, dtype_map)
    
    def _apply_dtypes(self, df: pd.DataFrame, dtype_map: Dict[str, type]) -> pd.DataFrame:
        """
        应用数据类型转换
        
        Parameters:
        -----------
        df : pd.DataFrame
            输入DataFrame
        dtype_map : Dict[str, type]
            数据类型映射
            
        Returns:
        --------
        pd.DataFrame : 类型转换后的DataFrame
        """
        for col, dtype in dtype_map.items():
            if col in df.columns:
                try:
                    if dtype == int:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                    elif dtype == float:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)
                    elif dtype == str:
                        df[col] = df[col].astype(str)
                except Exception as e:
                    logger.warning(f"列 {col} 类型转换失败: {str(e)}")
        
        return df
    
    def load(self):
        """
        加载所有输出数据文件
        """
        logger.info(f"开始加载数据集: {self.dataset_name}")
        
        # 构建数据集目录路径
        dataset_path = os.path.join(self.input_file_loc, self.dataset_name)
        
        # 定义文件配置
        file_configs = {
            'result': {
                'filename': 'opt_result.csv',
                'title_map': {
                    'day': 'day', 'plant_code': 'fact_id', 
                    'client_code': 'dealer_id', 'product_code': 'sku_id', 
                    'vehicle_id': 'vehicle_id', 'vehicle_type': 'vehicle_type',
                    'qty': 'qty'
                },
                'dtype_map': {
                    'day': int, 'fact_id': str, 'dealer_id': str, 
                    'sku_id': str, 'vehicle_id': int, 'vehicle_type': str, 'qty': int
                }
            },
            'details': {
                'filename': 'opt_details.csv',
                'title_map': {
                    'day': 'day', 'plant_code': 'fact_id', 
                    'client_code': 'dealer_id', 'product_code': 'sku_id', 
                    'vehicle_id': 'vehicle_id', 'vehicle_type': 'vehicle_type',
                    'qty': 'qty', 'standard_size': 'size', 'carry_standard_size': 'capacity',
                    'min_standard_size': 'min_load', 'occupied_size': 'occupied_size'
                },
                'dtype_map': {
                    'day': int, 'fact_id': str, 'dealer_id': str, 'sku_id': str,
                    'vehicle_id': int, 'vehicle_type': str, 'qty': int,
                    'size': float, 'capacity': int, 'min_load': int, 'occupied_size': float
                }
            },
            'summary': {
                'filename': 'opt_summary.csv',
                'title_map': {
                    'day': 'day', 'plant_code': 'fact_id', 
                    'client_code': 'dealer_id', 'vehicle_id': 'vehicle_id', 
                    'vehicle_type': 'vehicle_type', 'qty': 'qty', 
                    'carry_standard_size': 'capacity', 'min_standard_size': 'min_load', 
                    'occupied_size': 'occupied_size'
                },
                'dtype_map': {
                    'day': int, 'fact_id': str, 'dealer_id': str, 'vehicle_id': int,
                    'vehicle_type': str, 'qty': int, 'capacity': int, 
                    'min_load': int, 'occupied_size': float
                }
            },
            'non_fulfill': {
                'filename': 'non_fulfill.csv',
                'title_map': {
                    'client_code': 'dealer_id', 'product_code': 'sku_id', 'volume': 'non_fulfill_qty'
                },
                'dtype_map': {
                    'dealer_id': str, 'sku_id': str, 'non_fulfill_qty': int
                }
            },
            'sku_inv_left': {
                'filename': 'sku_inv_left.csv',
                'title_map': {
                    'plant_code': 'fact_id', 'product_code': 'sku_id', 'day': 'day', 'volume': 'inv_left'
                },
                'dtype_map': {
                    'fact_id': str, 'sku_id': str, 'day': int, 'inv_left': int
                }
            }
        }
        
        # 加载所有文件
        loaded_data = {}
        for key, config in file_configs.items():
            file_path = os.path.join(dataset_path, config['filename'])
            df = self._load_and_process_csv(
                file_path, 
                config['title_map'], 
                config['dtype_map']
            )
            loaded_data[key] = df
        
        # 保存DataFrame引用
        self.df_result = loaded_data['result']
        self.df_details = loaded_data['details']
        self.df_summary = loaded_data['summary']
        self.df_non_fulfill = loaded_data['non_fulfill']
        self.df_sku_inv_left = loaded_data['sku_inv_left']
        
        # 生成字典索引
        self._generate_indexes()
        
        # 验证数据完整性
        self._validate_data()
        
        logger.info("数据加载完成")
    
    def _generate_indexes(self):
        """生成用于快速查询的字典索引"""
        try:
            # 订单履行情况
            if not self.df_result.empty:
                self.order_fulfill = self.df_result.groupby(
                    ['day', 'fact_id', 'dealer_id', 'sku_id']
                )['qty'].sum().to_dict()
            
            # 车辆装载情况
            if not self.df_summary.empty:
                self.vehicle_load = self.df_summary.set_index(
                    ['day', 'fact_id', 'dealer_id', 'vehicle_id', 'vehicle_type']
                )['occupied_size'].to_dict()
            
            # 未满足需求
            if not self.df_non_fulfill.empty:
                self.non_fulfill = self.df_non_fulfill.set_index(
                    ['dealer_id', 'sku_id']
                )['non_fulfill_qty'].to_dict()
            
            # 剩余库存
            if not self.df_sku_inv_left.empty:
                self.sku_inv_left = self.df_sku_inv_left.set_index(
                    ['fact_id', 'sku_id', 'day']
                )['inv_left'].to_dict()
                
            logger.info("字典索引生成完成")
            
        except Exception as e:
            logger.error(f"生成字典索引失败: {str(e)}")
    
    def _validate_data(self):
        """验证数据完整性"""
        issues = []
        
        # 检查关键数据是否为空
        if self.df_result.empty:
            issues.append("结果数据(opt_result.csv)为空")
        
        if self.df_summary.empty:
            issues.append("汇总数据(opt_summary.csv)为空")
        
        # 检查数据一致性
        if not self.df_result.empty and not self.df_summary.empty:
            result_vehicles = set(self.df_result['vehicle_id'].unique())
            summary_vehicles = set(self.df_summary['vehicle_id'].unique())
            if result_vehicles != summary_vehicles:
                issues.append("结果数据和汇总数据中的车辆ID不一致")
        
        # 检查负值
        if not self.df_result.empty and (self.df_result['qty'] < 0).any():
            issues.append("发现负的配送数量")
        
        if not self.df_sku_inv_left.empty and (self.df_sku_inv_left['inv_left'] < 0).any():
            issues.append("发现负的库存量")
        
        # 输出验证结果
        if issues:
            logger.warning("数据验证发现问题:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("数据验证通过")
    
    def get_summary_stats(self) -> Dict[str, int]:
        """
        获取数据汇总统计
        
        Returns:
        --------
        Dict[str, int] : 统计信息
        """
        stats = {
            'total_deliveries': len(self.order_fulfill),
            'total_vehicles_used': len(self.vehicle_load),
            'unfulfilled_demands': len(self.non_fulfill),
            'inventory_records': len(self.sku_inv_left)
        }
        
        if not self.df_result.empty:
            stats['total_quantity_shipped'] = int(self.df_result['qty'].sum())
            stats['unique_vehicles'] = int(self.df_result['vehicle_id'].nunique())
        
        return stats