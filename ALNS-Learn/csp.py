import copy
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from alns import ALNS
from alns.accept import HillClimbing
from alns.select import RouletteWheel
from alns.stop import MaxIterations

SEED = 5432

OPTIMAL_BEAMS = 74

# The first line lists the numbers of lines for beam orders.
# The second line is the length of the available beams.
# Each following line is an order of (length, amount) tuples.
with open('data/640.csp') as file:
    data = file.readlines()

NUM_LINES = int(data[0])
BEAM_LENGTH = int(data[1])

# Beam to be cut from the available beams.
BEAMS = [
    int(length)
    for datum in data[-NUM_LINES:]
    for length, amount in [datum.strip().split()]
    for _ in range(int(amount))
]

print(f"Each available beam is of length: {BEAM_LENGTH}")
print(f"Number of beams to be cut (orders): {len(BEAMS)}")

# To use the ALNS meta-heuristic, we need to have destroy and repair operators that work on a proposed solution,
# and a way to describe such a solution in the first place. Let’s start with the solution state.
class CspState:
    """
    Solution state for the CSP problem. It has two data members, assignments and unassigned.
    Assignments is a list of lists, one for each beam in use.
    Each entry is another list, containing the ordered beams cut from this beam.
    Each such sublist must sum to at most BEAM_LENGTH.
    Unassigned is a list of ordered beams tha are not currently assigned to one of the available beams.
    """

    def __init__(self, assignments, unassigned=None):
        self.assignments = assignments
        self.unassigned = []

        if unassigned is not None:
            self.unassigned = unassigned

    def copy(self):
        """
        Helper method to ensure each solution state is immutable.
        """
        return CspState(
            copy.deepcopy(self.assignments), self.unassigned.copy()
        )

    def objective(self):
        """
        Computes the total number of beams in use.
        """
        return len(self.assignments)

    def plot(self):
        """
        Helper method to plot a solution
        """
        _, ax = plt.subplots(figsize=(12, 6))
        ax.barh(
            np.arange(len(self.assignments)),
            [sum(assignment) for assignment in self.assignments],
            height=1,
        )

        ax.set_xlim(right=BEAM_LENGTH)
        ax.set_yticks(np.arange(len(self.assignments), step=10))

        ax.margins(x=0, y=0)

        ax.set_xlabel("Usage")
        ax.set_ylabel("Beam (#)")

        plt.show()


def wastage(assignment):
    """
    Helper method that computes the wastage on a given beam assignment.
    """
    return BEAM_LENGTH - sum(assignment)

# Two destroy operators
# one is random_removal, another is worst_removal.
# Random removal randomly removes currently assigned beams, whereas worst removal removes those beams that are currently
# cut with the most waste. Both remove a fixed percentage of the current solution state, controlled by a degree of destruction parameter.
degree_of_destruction = 0.25

def beams_to_remove(num_beams):
    return int(num_beams * degree_of_destruction)

def random_removal(state, rng):
    """
    Iteratively removes randomly chosen beam assignments.
    """
    state = state.copy()

    for _ in range(beams_to_remove(state.objective())):
        idx = rng.integers(state.objective())
        state.unassigned.extend(state.assignments.pop(idx))
    return state

def worst_removal(state, rng):
    """
    Removes beams in decreasing order of wastage, such that the
    poorest assignment are removed first.
    """
    state = state.copy()

    # sort assignments by wastage, worst first
    state.assignments.sort(key=wastage, reverse=True)

    # removes the worst assignments
    for _ in range(beams_to_remove(state.objective())):
        state.unassigned.extend(state.assignments.pop(0))
    return state

# Two repair operators
# one is greedy_insert, another is minimal_wastage.
# The first considers each currently unassigned ordered beam, and finds the first beam this order may be inserted into.
# The second does something similar, but finds a beam where its insertion would result in the smallest beam wastage.
def greedy_insert(state, rng):
    """
    Inserts the unassigned beams greedily into the first fitting beam.
    Shuffles the unassigned beams before inserting.
    """
    rng.shuffle(state.unassigned)

    while len(state.unassigned) != 0:
        beam = state.unassigned.pop(0)

        for assignment in state.assignments:
            if beam <= wastage(assignment):
                assignment.append(beam)
                break
        else:
            state.assignments.append([beam])

    return state

def minimal_wastage(state, rng):
    """
    For every unassigned ordered beam, the operator determines
    which beam would minimize that beam's waste once the ordered beam is inserted.
    """
    def insertion_cost(assignment, beam):  # helper method for min
        if beam <= wastage(assignment):
            return wastage(assignment) - beam

        return float('inf')

    while len(state.unassigned) != 0:
        beam = state.unassigned.pop(0)

        assignment = min(
            state.assignments,
            key=partial(insertion_cost, beam=beam)
        )

        if beam <= wastage(assignment):
            assignment.append(beam)
        else:
            state.assignments.append([beam])

    return state


# Initial solution
rng = rnd.default_rng(SEED)

state = CspState([], BEAMS.copy())
init_sol = greedy_insert(state, rng)

print(f"Initial solution has objective value: {init_sol.objective()}")
init_sol.plot()

# Heuristic solution
alns = ALNS(rng)

alns.add_destroy_operator(random_removal)
alns.add_destroy_operator(worst_removal)

alns.add_repair_operator(greedy_insert)
alns.add_repair_operator(minimal_wastage)

accept = HillClimbing()
select = RouletteWheel(
    scores=[3, 2, 1, 0.5],
    decay=0.8,
    num_destroy=2,
    num_repair=2,
)
stop = MaxIterations(5_000)

result = alns.iterate(init_sol, select, accept, stop)
solution = result.best_state
objective = solution.objective()

_, ax = plt.subplots(figsize=(12, 6))
result.plot_objectives(ax=ax)

# Having obtained a reasonable solution, we now want to investigate each operator’s performance.
# This may be done via the plot_operator_counts() method on the result object, like below.
figure = plt.figure("operator_counts", figsize=(12, 6))
figure.subplots_adjust(bottom=0.15, hspace=0.5)
result.plot_operator_counts(figure, title="Operator diagnostics")

print("Heuristic solution has objective value: ", solution.objective())
solution.plot()

obj = solution.objective()
print(f"Number of beams used is {obj}, which is {obj - OPTIMAL_BEAMS} more than the optimal value {OPTIMAL_BEAMS}.")
