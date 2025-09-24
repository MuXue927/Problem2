import matplotlib.pyplot as plt
import numpy as np

# # 价格变化范围
# price_changes = np.linspace(0.8, 1.2, 10)
# essential_demand = np.ones(10) * 100  # 必需品需求稳定（低弹性）
# luxury_demand = 100 / price_changes  # 奢侈品需求较大弹性

# # 创建图表
# fig, ax = plt.subplots(figsize=(7, 5))
# ax.plot(price_changes, essential_demand, label="Essential Goods (Inelastic)", linestyle='-', color='g', linewidth=2)
# ax.plot(price_changes, luxury_demand, label="Luxury Goods (Elastic)", linestyle='--', color='m', linewidth=2)

# # 添加标签和标题
# ax.set_xlabel("Price Change (%)")
# ax.set_ylabel("Demand (Indexed)")
# ax.set_title("Price Elasticity of Demand")
# ax.legend()
# ax.grid(True)

# # 显示图表
# plt.show()
# fig.savefig(".//images//price_elasticity.svg", dpi=600)


# # Data for Inflation Theory: Headline vs. Underlying Inflation
# years = np.arange(2020, 2025, 1)
# headline_inflation = [3.5, 4.8, 6.1, 5.2, 2.8]  # Example headline inflation trend
# core_inflation = [3.2, 4.5, 5.3, 4.1, 3.5]  # Underlying inflation trend

# # Create the plot
# plt.figure(figsize=(8, 5))
# plt.plot(years, headline_inflation, marker='o', label="Headline Inflation", color='blue')
# plt.plot(years, core_inflation, marker='s', linestyle='dashed', label="Underlying Inflation", color='green')
# plt.axhline(3, color='r', linestyle='--', label="RBA Target (Upper Limit)")

# # Labels and title
# plt.title("Headline vs. Underlying Inflation (2020-2024)")
# plt.xlabel("Year")
# plt.ylabel("Inflation Rate (%)")
# plt.legend()
# plt.grid(True)

# # Save and display the chart
# inflation_chart_path = ".//images//inflation_trend.png"
# plt.savefig(inflation_chart_path)
# plt.show()


# # 图1: 货币政策与利率变化
# years = np.arange(2020, 2025, 1)
# interest_rates = [0.25, 0.1, 1.85, 3.1, 4.35]  # 假设利率走势

# plt.figure(figsize=(6, 4))
# plt.plot(years, interest_rates, marker='o', linestyle='-', label="Cash Rate (%)")
# plt.xlabel("Year")
# plt.ylabel("Interest Rate (%)")
# plt.title("Monetary Policy: Interest Rate Changes")
# plt.legend()
# plt.grid(True)
# plt.savefig(".//images//monetary_policy.png")  # 保存图像

# # 图2: 通货膨胀率 vs RBA 目标范围
# years = np.arange(2020, 2025, 1)
# inflation_rates = [0.9, 3.5, 7.0, 5.1, 2.8]  # 假设通胀率走势
# target_range = [2.5] * len(years)

# plt.figure(figsize=(6, 4))
# plt.plot(years, inflation_rates, marker='o', linestyle='-', label="Inflation Rate (%)", color="red")
# plt.axhline(y=2.5, color='green', linestyle='--', label="RBA Target (2-3%)")
# plt.xlabel("Year")
# plt.ylabel("Inflation Rate (%)")
# plt.title("Inflation Rate vs RBA Target")
# plt.legend()
# plt.grid(True)
# plt.savefig(".//images//inflation_target.png")


# # 1. Supply and Demand Curve
# # Create data for a simple supply and demand curve
# price = np.linspace(0, 10, 100)
# demand = 100 - 10 * price  # Demand curve (downward sloping)
# supply = 2 * price  # Supply curve (upward sloping)

# # Create the plot
# plt.figure(figsize=(8, 6))
# plt.plot(price, demand, label='Demand', color='blue')
# plt.plot(price, supply, label='Supply', color='red')
# plt.title('Supply and Demand Curves for Vapes')
# plt.xlabel('Price')
# plt.ylabel('Quantity')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('.//images//supply_demand_curve.png')
# plt.show()

# # 2. Price Elasticity of Demand (PED) Example
# # Creating a simple plot for Price Elasticity of Demand (Inelastic Demand)
# price_inelastic = np.linspace(1, 10, 100)
# quantity_inelastic = 100 / price_inelastic  # Inelastic demand (quantity does not change much with price)

# # Create the plot
# plt.figure(figsize=(8, 6))
# plt.plot(price_inelastic, quantity_inelastic, label='Inelastic Demand', color='green')
# plt.title('Price Elasticity of Demand (Inelastic)')
# plt.xlabel('Price')
# plt.ylabel('Quantity Demanded')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('.//images//inelastic_demand.png')
# plt.show()

# # 3. Market Failure Example (Shifting Demand Curve due to a Ban)
# # Create data for demand shifting due to external factors (e.g., government ban)
# price_failure = np.linspace(0, 10, 100)
# demand_no_ban = 100 - 10 * price_failure  # Initial demand curve (without ban)
# demand_with_ban = 80 - 10 * price_failure  # Demand curve shifts left due to ban

# # Create the plot
# plt.figure(figsize=(8, 6))
# plt.plot(price_failure, demand_no_ban, label='Demand (No Ban)', color='orange')
# plt.plot(price_failure, demand_with_ban, label='Demand (With Ban)', color='purple', linestyle='--')
# plt.title('Market Failure: Effect of Vaping Ban on Demand')
# plt.xlabel('Price')
# plt.ylabel('Quantity')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('.//images//market_failure.png')


# # Create a figure for the first economic concept: Headline vs. Underlying Inflation
# years = np.arange(2020, 2025, 1)
# headline_inflation = [3.5, 4.8, 6.1, 5.2, 2.8]  # Example headline inflation trend
# core_inflation = [3.2, 4.5, 5.3, 4.1, 3.5]  # Underlying inflation trend

# plt.figure(figsize=(8, 5))
# plt.plot(years, headline_inflation, marker='o', label="Headline Inflation", color='blue')
# plt.plot(years, core_inflation, marker='s', linestyle='dashed', label="Underlying Inflation", color='green')
# plt.axhline(3, color='r', linestyle='--', label="RBA Target (Upper Limit)")

# # Labels and title
# plt.title("Headline vs. Underlying Inflation (2020-2024)")
# plt.xlabel("Year")
# plt.ylabel("Inflation Rate (%)")
# plt.legend()
# plt.grid(True)

# # Save the figure
# inflation_chart_path = ".//images//inflation_trend.png"
# plt.savefig(inflation_chart_path)
# plt.show()

# # Data for Inflation Theory: Headline vs. Service Inflation
# years = np.arange(2020, 2025, 1)
# headline_inflation = [3.5, 4.8, 6.1, 5.2, 2.8]  # Headline inflation trend
# service_inflation = [3.8, 4.2, 5.0, 4.8, 4.6]  # Service sector inflation trend

# # Create the plot
# plt.figure(figsize=(8, 5))
# plt.bar(years - 0.15, headline_inflation, width=0.3, label="Headline Inflation", color='blue', alpha=0.7)
# plt.bar(years + 0.15, service_inflation, width=0.3, label="Service Inflation", color='orange', alpha=0.7)

# # Labels and title
# plt.title("Headline vs. Service Inflation (2020-2024)")
# plt.xlabel("Year")
# plt.ylabel("Inflation Rate (%)")
# plt.xticks(years)
# plt.legend()
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# # Save the figure
# inflation_service_chart_path = ".//images//inflation_service_trend.png"
# plt.savefig(inflation_service_chart_path)
# plt.show()


# # Inflation and Purchasing Power
# plt.figure(figsize=(6, 4))
# x = np.linspace(0, 10, 100)
# inflation = np.exp(x / 10)  # Exponential increase to represent inflation
# real_income = 10 / inflation  # Inverse relationship with purchasing power
# plt.plot(x, inflation, label="Inflation Rate", color='r')
# plt.plot(x, real_income, label="Real Income", color='b')
# plt.title("Inflation and Purchasing Power")
# plt.xlabel("Time (Years)")
# plt.ylabel("Index Value")
# plt.legend()
# inflation_img = ".//images//inflation_purchasing_power.png"
# plt.savefig(inflation_img)


# # Interest Rates and Borrowing Costs
# plt.figure(figsize=(6, 4))
# interest_rates = np.linspace(1, 10, 100)
# borrowing_costs = np.sqrt(interest_rates)  # Higher interest rates lead to higher borrowing costs
# plt.plot(interest_rates, borrowing_costs, label="Borrowing Costs", color='g')
# plt.title("Interest Rates and Borrowing Costs")
# plt.xlabel("Interest Rate (%)")
# plt.ylabel("Borrowing Cost Index")
# plt.legend()
# interest_img = ".//images//interest_rates_borrowing_costs.png"
# plt.savefig(interest_img)


# # Price Elasticity of Demand
# plt.figure(figsize=(6, 4))
# prices = np.linspace(1, 10, 100)
# inelastic_demand = 10 - 0.5 * prices  # Small change in quantity demanded
# elastic_demand = 10 - 2.5 * prices  # Large change in quantity demanded
# plt.plot(prices, inelastic_demand, label="Inelastic Demand", color='b')
# plt.plot(prices, elastic_demand, label="Elastic Demand", color='r')
# plt.title("Price Elasticity of Demand")
# plt.xlabel("Price Level")
# plt.ylabel("Quantity Demanded")
# plt.legend()
# elasticity_img = ".//images//price_elasticity_demand.png"
# plt.savefig(elasticity_img)


# # 创建三个经济概念的相关图表

# # 1. 失业类型：摩擦性、结构性、周期性失业趋势图
# fig1, ax1 = plt.subplots()
# x = np.arange(2019, 2025, 1)
# frictional_unemployment = [3.5, 3.7, 3.6, 3.8, 4.0, 4.1]  # 摩擦性失业
# structural_unemployment = [2.5, 2.6, 2.7, 3.0, 3.2, 3.4]  # 结构性失业
# cyclical_unemployment = [1.0, 1.5, 4.5, 3.0, 2.5, 2.0]  # 周期性失业

# ax1.plot(x, frictional_unemployment, label="Frictional Unemployment", marker='o')
# ax1.plot(x, structural_unemployment, label="Structural Unemployment", marker='s')
# ax1.plot(x, cyclical_unemployment, label="Cyclical Unemployment", marker='^')
# ax1.set_xlabel("Year")
# ax1.set_ylabel("Unemployment Rate (%)")
# ax1.set_title("Types of Unemployment Trends (2019-2025)")
# ax1.legend()
# plt.grid()

# # 保存第一张图
# plt.savefig(".//images//unemployment_types_trends.png")

# # 2. 货币政策影响：利率 vs. 失业率
# fig2, ax2 = plt.subplots()
# interest_rates = [1.0, 0.5, 0.1, 2.0, 3.5, 4.0]  # 利率水平
# unemployment_rates = [5.0, 4.8, 4.5, 4.7, 4.9, 5.2]  # 失业率

# ax2.plot(interest_rates, unemployment_rates, marker='o', linestyle='-')
# ax2.set_xlabel("Interest Rate (%)")
# ax2.set_ylabel("Unemployment Rate (%)")
# ax2.set_title("Impact of Monetary Policy on Unemployment")
# plt.grid()

# # 保存第二张图
# plt.savefig(".//images//monetary_policy_vs_unemployment.png")

# # 3. 劳动力市场技能错配（高技能 vs. 低技能失业率）
# fig3, ax3 = plt.subplots()
# years = np.arange(2020, 2025, 1)
# high_skill_unemployment = [2.5, 3.0, 3.8, 4.2, 4.5]  # 高技能人才失业率
# low_skill_unemployment = [5.0, 5.2, 5.5, 6.0, 6.3]  # 低技能人才失业率

# ax3.plot(years, high_skill_unemployment, label="High-Skill Unemployment", marker='o', linestyle='-')
# ax3.plot(years, low_skill_unemployment, label="Low-Skill Unemployment", marker='s', linestyle='-')
# ax3.set_xlabel("Year")
# ax3.set_ylabel("Unemployment Rate (%)")
# ax3.set_title("Labour Market Skill Mismatch (2020-2025)")
# ax3.legend()
# plt.grid()

# # 保存第三张图
# plt.savefig(".//images//labour_market_skill_mismatch.png")

# plt.show()


# # 创建图表 1 - 货币政策对通胀和失业率的影响
# fig, ax1 = plt.subplots(figsize=(7,5))

# # X轴表示时间（假设为季度）
# quarters = np.arange(1, 11)

# # 假设利率上升后，通胀下降，失业率上升
# inflation_rates = np.linspace(5, 2.8, 10)  # 通胀下降
# unemployment_rates = np.linspace(3.5, 4.5, 10)  # 失业率上升

# ax1.set_xlabel('Time (Quarters)')
# ax1.set_ylabel('Inflation Rate (%)', color='tab:red')
# ax1.plot(quarters, inflation_rates, 'r-o', label="Inflation Rate")
# ax1.tick_params(axis='y', labelcolor='tab:red')

# ax2 = ax1.twinx()
# ax2.set_ylabel('Unemployment Rate (%)', color='tab:blue')
# ax2.plot(quarters, unemployment_rates, 'b-s', label="Unemployment Rate")
# ax2.tick_params(axis='y', labelcolor='tab:blue')

# plt.title("Impact of Interest Rate on Inflation & Unemployment")
# plt.savefig(".//images//interest_rate_on_inflation_unemployment.png")

# # 创建第二张图 - 需求拉动 vs 成本推动通胀
# fig, ax = plt.subplots(figsize=(7,5))

# categories = ['Demand-Pull Inflation', 'Cost-Push Inflation']
# values = [55, 45]  # 假设数据

# ax.bar(categories, values, color=['orange', 'purple'])
# ax.set_ylabel("Contribution to Inflation (%)")
# ax.set_title("Types of Inflation in Australia")
# plt.savefig(".//images//types_of_inflation_in_australia.png")

# # 创建第三张图 - 菲利普斯曲线（Phillips Curve）
# fig, ax = plt.subplots(figsize=(7,5))

# unemployment = np.linspace(3, 6, 10)
# inflation = 5 - 0.5 * unemployment  # 假设的菲利普斯曲线关系

# ax.plot(unemployment, inflation, 'g-o', label="Phillips Curve")
# ax.set_xlabel("Unemployment Rate (%)")
# ax.set_ylabel("Inflation Rate (%)")
# ax.set_title("Phillips Curve: Inflation vs Unemployment")
# ax.legend()
# plt.savefig(".//images//phillips_curve.png")


# 创建图表 1：供需曲线（Supply and Demand）
plt.figure(figsize=(6, 4))
q = np.linspace(0, 10, 100)
demand = 10 - q  # 需求曲线
supply_before = q  # 供应曲线（禁令前）
supply_after = q + 3  # 供应曲线（禁令后，左移）

plt.plot(q, demand, label="Demand Curve", color='blue')
plt.plot(q, supply_before, label="Supply Curve (Before Ban)", color='green')
plt.plot(q, supply_after, label="Supply Curve (After Ban)", color='red', linestyle='dashed')

plt.xlabel("Quantity of Vapes")
plt.ylabel("Price")
plt.title("Impact of Vaping Ban on Supply and Demand")
plt.legend()
plt.grid(True)
plt.savefig(".//images//supply_demand_vape_ban.png")
plt.show()

# 创建图表 2：黑市价格机制（Black Market Pricing）
plt.figure(figsize=(6, 4))
quantity = np.array([1, 2, 3, 4, 5])
price_legal = np.array([20, 18, 16, 15, 14])
price_black_market = np.array([50, 45, 40, 38, 35])

plt.plot(quantity, price_legal, marker='o', label="Legal Market Price", color='green')
plt.plot(quantity, price_black_market, marker='s', label="Black Market Price", color='red', linestyle='dashed')

plt.xlabel("Quantity of Vapes")
plt.ylabel("Price (AUD)")
plt.title("Black Market Pricing Mechanism vs Legal Market")
plt.legend()
plt.grid(True)
plt.savefig(".//images//black_market_pricing.png")
plt.show()

# 创建图表 3：市场失灵（Market Failure - Public Health Risk）
labels = ["Regulated Market", "Black Market"]
risks = [20, 80]  # 假设黑市产品带来的健康风险更高

plt.figure(figsize=(6, 4))
plt.bar(labels, risks, color=['blue', 'red'])
plt.ylabel("Health Risk Level (%)")
plt.title("Market Failure: Public Health Risk Due to Black Market")

plt.savefig(".//images//market_failure_health_risk.png")
plt.show()
