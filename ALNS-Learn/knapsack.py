import matplotlib.pyplot as plt
import numpy as np
from mabwiser.mab import LearningPolicy
from alns import ALNS
from alns.accept import *
from alns.select import *
from alns.stop import *


SEED = 42
np.random.seed(SEED)

# Define the problem parameters
n = 100  # number of items
p = np.random.randint(1, 100, size=n)  # profit of each item
w = np.random.randint(10, 50, size=n)  # weight of each item
W = 1_000  # maximum weight of the knapsack

# Percentage of items to remove in each iteration
destroy_rate = 0.25

# Define the solution state
class KnapsackState:
    """
    Solution class for the 0/1 knapsack problem. It stores the current
    solution as a vector of binary variables, one for each item.
    """

    def __init__(self, x: np.ndarray):
        self.x = x

    def objective(self) -> int:
        # Negative p since ALNS expects a minimization problem.
        return -p @ self.x

    def weight(self) -> int:
        return w @ self.x

# Two destroy operators:
# one is a random destroy operator, which removes items from the knapsack at random.
# another operator that removes items based on their relative merits, that is, for an
# item i currently in the knapsack, it removes those whose p_i / w_i values are smallest.

def to_destroy(state: KnapsackState) -> int:
    return int(destroy_rate * state.x.sum())

def random_remove(state: KnapsackState, rng):
    probs = state.x / state.x.sum()
    to_remove = rng.choice(np.arange(n), size=to_destroy(state), p=probs)

    assignments = state.x.copy()
    assignments[to_remove] = 0

    return KnapsackState(x=assignments)

def worst_remove(state: KnapsackState, rng):
    merit = state.x * p / w
    by_merit = np.argsort(-merit)
    by_merit = by_merit[by_merit > 0]
    to_remove = by_merit[: to_destroy(state)]

    assignments = state.x.copy()
    assignments[to_remove] = 0

    return KnapsackState(x=assignments)

# Only the random repair operator
def random_repair(state: KnapsackState, rng):
    unselected = np.argwhere(state.x == 0)
    rng.shuffle(unselected)

    while True:
        can_insert = w[unselected] <= W - state.weight()
        unselected = unselected[can_insert]

        if len(unselected) != 0:
            insert, unselected = unselected[0], unselected[1:]
            state.x[insert] = 1
        else:
            return state

# ALNS
def make_alns() -> ALNS:
    rng = np.random.default_rng(SEED)
    alns = ALNS(rng)

    alns.add_destroy_operator(random_remove)
    alns.add_destroy_operator(worst_remove)

    alns.add_repair_operator(random_repair)

    return alns

# Terrible - but simple - first solution, where only the first item is selected.
init_sol = KnapsackState(np.zeros(n))
init_sol.x[0] = 1

# Operator selection schemes
# use the HillClimbing acceptance criterion, which only accepts better soutions.
accept = HillClimbing()

# Roulette wheel
# The RouletteWheel scheme updates operator weights as a convex combination of
# the current weight, and the new score.
select = RouletteWheel(
    scores=[5, 2, 1, 0.5], decay=0.8, num_destroy=2, num_repair=1
)

alns = make_alns()
res = alns.iterate(init_sol, select, accept, MaxIterations(10_000))

print(f"Found solution with objective {-res.best_state.objective()}")

_, ax = plt.subplots(figsize=(12, 6))
res.plot_objectives(ax=ax, lw=2)
plt.show()

# Segmented roulette wheel
# The RouletteWheel scheme continuously updates the weights of the destroy and repair operators.
# As a consequence, it might overlook that different operators are more effective in the neighbourhood of different solutions.
# The SegmentedRouletteWheel scheme attempts to fix this, by fixing the operator weights w_i
# for a number of iterations (the segment length).
select = SegmentedRouletteWheel(
    scores=[5, 2, 1, 0.5],
    decay=0.8,
    seg_length=500,
    num_destroy=2,
    num_repair=1,
)

alns = make_alns()
res = alns.iterate(init_sol, select, accept, MaxIterations(10_000))

print(f"Found solution with objective {-res.best_state.objective()}")

_, ax = plt.subplots(figsize=(12, 6))
res.plot_objectives(ax=ax, lw=2)
plt.show()

# alpha-UCB
# The alpha-UCB scheme is an upper confidence bound bandit algorithm that learns good
# (destroy, repair) operator pairs, and plays those more often during the search.
# Typically, alpha <= 0.1 is a good choice.
select = AlphaUCB(
    scores=[5, 2, 1, 0.5],
    alpha=0.05,
    num_destroy=2,
    num_repair=1
)

alns = make_alns()
res = alns.iterate(init_sol, select, accept, MaxIterations(10_000))

print(f"Found solution with objective {-res.best_state.objective()}")

_, ax = plt.subplots(figsize=(12, 6))
res.plot_objectives(ax=ax, lw=2)
plt.show()

# More advanced bandit algorithms
# Operator selection can be seen as a multi-armed-bandit problem. Each operator pair is a bandit arm,
# and the reward for each arm corresponds to the evaluation outcome depending on the score array.
select = MABSelector(
    scores=[5, 2, 1, 0.5],
    num_destroy=2,
    num_repair=1,
    learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15),
)

alns = make_alns()
res = alns.iterate(init_sol, select, accept, MaxIterations(10_000))

print(f"Found solution with objective {-res.best_state.objective()}")

_, ax = plt.subplots(figsize=(12, 6))
res.plot_objectives(ax=ax, lw=2)
plt.show()

# Contextual bandit algorithms
# Some MABWiser bandit algorithms require a context vector when making an operator selection choice.
# Here we monkey-patch our existing KnapsackState to conform to the ContextualState protocol.
def get_knapsack_context(self: KnapsackState):
    num_items = np.count_nonzero(self.x)
    avg_weight = self.weight() / num_items
    return np.array([self.weight(), num_items, avg_weight])

KnapsackState.get_context = get_knapsack_context

select = MABSelector(
    scores=[5, 2, 1, 0.5],
    num_destroy=2,
    num_repair=1,
    learning_policy=LearningPolicy.LinGreedy(epsilon=0.15),
)

alns = make_alns()
res = alns.iterate(init_sol, select, accept, MaxIterations(10_000))

print(f"Found solution with objective {-res.best_state.objective()}")

_, ax = plt.subplots(figsize=(12, 6))
res.plot_objectives(ax=ax, lw=2)
plt.show()

# Acceptance criteria
# We have just looked at the different weight schemes, using a fixed acceptance criterion. Now we flip this around:
# we fix an operator selection scheme, and look at several acceptance criteria the alns package offers.
select = SegmentedRouletteWheel(
    scores=[5, 2, 1, 0.5],
    decay=0.8,
    seg_length=500,
    num_destroy=2,
    num_repair=1,
)

# Hill climbing
# This acceptance criterion only accepts better solutions.
# It was used in the examples explaining the operator selection schemes, so we will not repeat it here.

# Record-to-record travel
# This criterion accepts solutions when the improvement meets some updating threshold.
# In particular, consider the current best solution s^* with objective f(s^*). A new candidate
# solution s^c is accepted if the improvement f(s^c) - f(s^*) is smaller than some updating threshold t.
# There are two ways in which this update can take place:
# liner: the threshold is updated linearly, as t= t- u.
# exponential: the threshold is updated exponentially, as t= t * u.
# Finally, the threshold t cannot become smaller than a minimum value, the end threshold.
accept = RecordToRecordTravel(
    start_threshold=255,
    end_threshold=5,
    step=250 / 10_000,
    method="linear"
)

alns = make_alns()
res = alns.iterate(init_sol, select, accept, MaxIterations(10_000))

print(f"Found solution with objective {-res.best_state.objective()}")

_, ax = plt.subplots(figsize=(12, 6))
res.plot_objectives(ax=ax, lw=2)
plt.show()

# Simulated annealing
# This criterion accepts solutions when the scaled probability is bigger than some random number,
# using an updating temperature that drives the probability down. It is very similar to
# the RecordToRecordTravel criterion, but uses a different acceptance scheme.
# In particular, a temperature is used, rather than a threshold, and the candidate s^c is compared
# against the current solution s, rather than the current best solution s*.
# The acceptance probability is calculated as
# exp(-(f(s^c) - f(s)) / t), where t is the current temperature.
accept = SimulatedAnnealing(
    start_temperature=1_000,
    end_temperature=1,
    step=1 - 1e-3,
    method="exponential",
)

alns = make_alns()
res = alns.iterate(init_sol, select, accept, MaxIterations(10_000))

print(f"Found solution with objective {-res.best_state.objective()}")

_, ax = plt.subplots(figsize=(12, 6))
res.plot_objectives(ax=ax, lw=2)
plt.show()

# Rather than a fixed number of iterations, we can also fix the runtime,
# and allow as many iterations as fit in that timeframe.
accept = SimulatedAnnealing(
    start_temperature=1_000,
    end_temperature=1,
    step=1 - 1e-3,
    method="exponential",
)

alns = make_alns()
res = alns.iterate(init_sol, select, accept, MaxRuntime(60))  # one minute

print(f"Found solution with objective {-res.best_state.objective()}")

_, ax = plt.subplots(figsize=(12, 6))
res.plot_objectives(ax=ax, lw=2)
plt.show()
