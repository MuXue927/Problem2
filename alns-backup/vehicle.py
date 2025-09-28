# 定义Vehicle类，用于模型初始可行解的构造
# Vehicle类需要提供的功能包括
# 1. 明确是在哪个周期 t 内, 哪个生产基地 i 为哪个经销商 j 使用哪种车型 f
# 2. 需要能够计算车辆上装载的商品的体积
# 3. 需要判断当前车辆是否满载
from InputDataALNS import DataALNS

from dataclasses import dataclass, field
from typing import Dict, Tuple

# dataclasses -- Data Classes
# This module provides a decorator and functions for automatically adding generated special methods such as
# __init__() and __repr__() to user-defined classes.


@dataclass
class Vehicle:
    fact_id: str
    dealer_id: str
    type: str
    day: int
    data: DataALNS

    id: int = field(init=False)
    max_capacity: int = field(init=False)
    capacity: int = field(init=False)
    cargo: Dict[Tuple[str, int], int] = field(default_factory=dict)

    _id_counter = 0  # 类变量

    def __post_init__(self):
        self.id = Vehicle._id_counter
        Vehicle._id_counter += 1
        self.max_capacity = self.data.veh_type_cap[self.type]
        self.capacity = self.max_capacity
        self.cargo = {}

    def __eq__(self, other):
        if not isinstance(other, Vehicle):
            return False
        return (self.fact_id == other.fact_id and
                self.dealer_id == other.dealer_id and
                self.type == other.type and
                self.day == other.day and
                self.id == other.id and
                self.cargo == other.cargo)

    def __hash__(self):
        return hash((self.fact_id, self.dealer_id, self.type, self.day, self.id))

    def __repr__(self):
        return (f"Vehicle(id={self.id}, fact_id={self.fact_id}, dealer_id={self.dealer_id}, "
                f"type={self.type}, day={self.day}, capacity={self.capacity}, cargo={self.cargo})")

    def get_avail_space(self) -> int:
        return self.capacity

    def is_full(self) -> bool:
        return self.capacity <= 0

    def load(self, sku_id: str, num: int) -> int:
        if num <= 0:
            return num
        if sku_id not in self.data.sku_sizes:
            raise ValueError(f"SKU {sku_id} not found in data.sku_sizes")
        sku_size = self.data.sku_sizes[sku_id]
        num_loaded = min(num, self.capacity // sku_size)
        
        if num_loaded > 0:
            self.cargo[(sku_id, self.day)] = self.cargo.get((sku_id, self.day), 0) + num_loaded
            self.capacity -= num_loaded * sku_size
        
        return num - num_loaded  # 返回未装载数量

    def clear(self):
        self.capacity = self.max_capacity
        self.cargo.clear()

    def is_empty(self) -> bool:
        return not self.cargo
