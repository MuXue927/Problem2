import pytest
from ALNSCode.InputDataALNS import DataALNS
from ALNSCode.alnsopt import SolutionState, veh_loading
from ALNSCode.vehicle import Vehicle


def build_minimal_data():
    """构造一个最小的 DataALNS 实例，用于单元测试。
    我们不调用 load()，而是手工设置必要字段。
    """
    data = DataALNS(input_file_loc="", output_file_loc="", dataset_name="test")
    
    # 规划期为3天（horizons=3）以便观察传播
    data.horizons = 3

    # SKU 尺寸
    data.sku_sizes = {"skuA": 10}

    # 每日生产量: day=1有50, day=2,3无生产
    data.sku_prod_each_day = {("plant1", "skuA", 1): 50}

    # 期初库存: plant1 对 skuA 在 day=0 有 100
    data.historical_s_ikt = {("plant1", "skuA", 0): 100}

    # plant->sku 可提供集合
    data.plants = {"plant1"}
    data.skus_plant = {"plant1": {"skuA"}}
    
    # dealer/需求相关（construct_supply_chain 需要 skus_dealer）
    data.dealers = {"dealer1"}
    data.skus_dealer = {"dealer1": {"skuA"}}
    data.all_skus = {"skuA"}
    data.all_veh_types = {"van"}

    # 车辆类型容量
    data.veh_type_cap = {"van": 100}
    data.veh_type_min_load = {"van": 0}
    data.veh_type_cost = {"van": 0}

    # 其他必要占位
    data.plant_inv_limit = {"plant1": 10000}
    data.demands = {}
    return data


def test_compute_inventory_and_incremental_update():
    data = build_minimal_data()
    state = SolutionState(data)

    # 初始时 state.s_ikt 只包含 day=0 的期初库存，compute_inventory 会填充 day>0
    assert state.s_ikt.get(("plant1", "skuA", 0)) == 100
    assert not state.s_initialized

    # 做一次完整重算，得到基线 s_ikt
    state.compute_inventory()
    assert state.s_initialized

    # 基线计算: day=1: prev(100)+prod(50)-shipped(0)=150
    assert state.s_ikt[("plant1", "skuA", 1)] == 150
    # day=2/3 没有生产或发货，因此应该继承前一天
    assert state.s_ikt[("plant1", "skuA", 2)] == 150
    assert state.s_ikt[("plant1", "skuA", 3)] == 150

    # 现在创建一辆车并装载 30 个单位的 skuA（每个单位体积 10，车辆容量 100，最多可装 10 个）
    veh = Vehicle("plant1", "dealer1", "van", 1, data)

    # orders 表示此经销商对 skuA 的需求（只需触发装载循环）
    orders = {"skuA": 30}

    # 调用 veh_loading 来进行装载，内部会做增量更新：把 load_qty 从 day=1 到 horizons 减去
    res = veh_loading(state, veh, orders)
    assert res is True

    # 车辆由于每件体积 10，车辆容量100，每辆车最多装 10 个单位。
    # veh_loading 会自动生成新的车辆以满足剩余订单量，因此本次总共装载 30 个单位，
    # 因此 s_ikt 在 day=1..3 都应该减少 30
    assert state.s_ikt[("plant1", "skuA", 1)] == 150 - 30
    assert state.s_ikt[("plant1", "skuA", 2)] == 150 - 30
    assert state.s_ikt[("plant1", "skuA", 3)] == 150 - 30

    # 另外，应该有多辆车被加入到 state.vehicles 来完成这 30 个单位的装载
    loaded_qty = sum(q for v in state.vehicles for q in v.cargo.values())
    assert loaded_qty == 30
    
    veh_nums = len(state.vehicles)
    assert veh_nums >= 3  # 至少3辆车，因为每辆车最多装10个单位