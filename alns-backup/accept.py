from InputDataALNS import DataALNS
from optutility import LogPrinter
import time

log_printer = LogPrinter(time.time())

# 创建一个自定义的接受准则，优先接受满足需求的解但不完全禁止不满足需求的解
class DemandConstraintAccept:
    def __init__(self, accept_criterion, data: DataALNS, penalty_factor=1000.0):
        """
        自定义接受准则，通过惩罚不满足需求的解来引导搜索
        
        Parameters:
        -----------
        accept_criterion : AcceptanceCriterion
            基础接受准则（如模拟退火）
        data : DataALNS
            数据实例
        penalty_factor : float
            对不满足需求的解的惩罚因子
        """
        self.accept_criterion = accept_criterion
        self.data = data
        self.penalty_factor = penalty_factor
        
        # 统计信息
        self.total_evaluations = 0
        self.feasible_accepted = 0
        self.infeasible_accepted = 0
        self.feasible_rejected = 0
        self.infeasible_rejected = 0
        
    def __call__(self, rng, best, current, candidate):
        """
        接受准则的主要逻辑
        
        Parameters:
        -----------
        rng : numpy.random.Generator
            随机数生成器
        best : SolutionState
            迄今为止的最优解
        current : SolutionState
            当前解
        candidate : SolutionState
            候选解
            
        Returns:
        --------
        bool : 是否接受候选解
        """
        self.total_evaluations += 1
        
        # 使用统一的验证方法检查候选解的可行性
        is_feasible, violations = candidate.validate()
        
        # 计算调整后的目标函数值（对不可行解添加惩罚）
        adjusted_objective = candidate.objective()
        if not is_feasible:
            # 计算违反约束的惩罚
            unmet_demands = violations.get('unmet_demand', [])
            total_unmet = sum(d['demand'] - d['shipped'] for d in unmet_demands)
            adjusted_objective += self.penalty_factor * total_unmet
        
        # 创建临时解对象用于基础接受准则判断
        class AdjustedSolution:
            def __init__(self, original_solution, adjusted_obj):
                self._original = original_solution
                self._adjusted_obj = adjusted_obj
            
            def objective(self):
                return self._adjusted_obj
            
            def __getattr__(self, name):
                return getattr(self._original, name)
        
        adjusted_candidate = AdjustedSolution(candidate, adjusted_objective)
        
        # 使用基础接受准则判断（考虑惩罚后的目标值）
        accept = self.accept_criterion(rng, best, current, adjusted_candidate)
        
        # 更新统计信息
        if accept:
            if is_feasible:
                self.feasible_accepted += 1
            else:
                self.infeasible_accepted += 1
        else:
            if is_feasible:
                self.feasible_rejected += 1
            else:
                self.infeasible_rejected += 1
        
        # 每100次评估输出一次统计信息
        if self.total_evaluations % 100 == 0:
            self._print_statistics(is_feasible, violations if not is_feasible else None)
        
        return accept
    
    def _print_statistics(self, current_feasible, violations=None):
        """
        打印统计信息
        
        Parameters:
        -----------
        current_feasible : bool
            当前解是否可行
        violations : dict, optional
            违反约束的详细信息
        """
        total = self.total_evaluations
        accept_rate = (self.feasible_accepted + self.infeasible_accepted) / total
        feasible_rate = (self.feasible_accepted + self.feasible_rejected) / total
        
        log_printer.print(f"=== Acceptance Statistics (after {total} evaluations) ===")
        log_printer.print(f"Overall accept rate: {accept_rate:.2%}")
        log_printer.print(f"Feasible solutions: {feasible_rate:.2%}")
        log_printer.print(f"Feasible accepted: {self.feasible_accepted}, rejected: {self.feasible_rejected}")
        log_printer.print(f"Infeasible accepted: {self.infeasible_accepted}, rejected: {self.infeasible_rejected}")
        
        if not current_feasible and violations:
            unmet_demands = violations.get('unmet_demand', [])
            if unmet_demands:
                total_unmet = sum(d['demand'] - d['shipped'] for d in unmet_demands)
                log_printer.print(f"Current solution: {len(unmet_demands)} unmet demands, total unmet: {total_unmet}")
        
        log_printer.print("=" * 50)
    
    def get_statistics(self):
        """
        获取统计信息
        
        Returns:
        --------
        dict : 统计信息字典
        """
        if self.total_evaluations == 0:
            return {}
        
        return {
            'total_evaluations': self.total_evaluations,
            'overall_accept_rate': (self.feasible_accepted + self.infeasible_accepted) / self.total_evaluations,
            'feasible_accept_rate': self.feasible_accepted / (self.feasible_accepted + self.feasible_rejected) if (self.feasible_accepted + self.feasible_rejected) > 0 else 0,
            'infeasible_accept_rate': self.infeasible_accepted / (self.infeasible_accepted + self.infeasible_rejected) if (self.infeasible_accepted + self.infeasible_rejected) > 0 else 0,
            'feasible_solutions_ratio': (self.feasible_accepted + self.feasible_rejected) / self.total_evaluations,
            'penalty_factor': self.penalty_factor
        }