"""
Day 1 projects
Based on your card, you will implement one of these:

Ace (or 9) - initialize_individual
King (or 8) - make_test_cases function and Individual's evaluate_individual method.
Queen - tournament_selection
Jack - mutation
10 - crossover
"""

import operator, random, math, copy

MAX_FLOAT = 1e12

def safe_division(numerator, denominator):
    """Divides numerator by denominator. If denominator is 0, returns
    MAX_FLOAT as an approximate of infinity."""
    if abs(denominator) <= 1 / MAX_FLOAT:
        return MAX_FLOAT
    return numerator / denominator

def safe_exp(power):
    """Takes e^power. If this results in a math overflow, or is greater
    than MAX_FLOAT, instead returns MAX_FLOAT"""
    try:
        result = math.exp(power)
        if result > MAX_FLOAT:
            return MAX_FLOAT
        return result
    except OverflowError:
        return MAX_FLOAT

FUNCTION_DICT = {"+": operator.add,
                 "-": operator.sub,
                 "*": operator.mul,
                 "/": safe_division,
                 "exp": safe_exp,
                 "sin": math.sin,
                 "cos": math.cos}
FUNCTION_ARITIES = {"+": 2,
                    "-": 2,
                    "*": 2,
                    "/": 2,
                    "exp": 1,
                    "sin": 1,
                    "cos": 1}

FUNCTIONS = list(FUNCTION_DICT.keys())

VARIABLES = ["x", "y"]
POPULATION_SIZE = 100
MAX_GENERATIONS = 100
TOURNAMENT_SIZE = 5
THRESHOLD = 0.5


class FunctionNode:
    """Internal nodes that contain Functions."""

    def __init__(self, function_symbol, children):
        self.function_symbol = function_symbol
        self.function = FUNCTION_DICT[self.function_symbol]
        self.children = children

    def __str__(self):
        result = "({}".format(self.function_symbol)
        for child in self.children:
            result += " " + str(child)
        result += ")"
        return result

    def eval(self, variable_assignments):
        """Evaluates node given a dictionary of variable assignments."""

        # Trying to catch a hard to reproduce error
        try:
            # Calculate values of children nodes
            children_results = [child.eval(variable_assignments) for child in self.children]

            # Apply function to children_results. * unpacks the list of results into
            # arguments to self.function.
            return self.function(*children_results)
        except ValueError as e:
            print("Weird value error:", e)
            print("Node causing it:", self)
            raise

    def tree_depth(self):
        """Returns the total depth of tree rooted at this node"""
        children_depths = [child.tree_depth() for child in self.children]
        return 1 + max(children_depths)

    def size_of_subtree(self):
        """Gives the size of the subtree of this node, in number of nodes."""
        children_sizes = [child.size_of_subtree() for child in self.children]
        return 1 + sum(children_sizes)


class TerminalNode:
    """Leaf nodes that contain terminals."""

    def __init__(self, terminal):
        self.terminal = terminal

    def __str__(self):
        return str(self.terminal)

    def eval(self, variable_assignments):
        """Evaluates node given a dictionary of variable assignments."""

        if self.terminal in variable_assignments:
            return variable_assignments[self.terminal]

        return self.terminal

    def tree_depth(self):
        """Returns the total depth of tree rooted at this node. Since terminal,
        just is 0."""
        return 0

    def size_of_subtree(self):
        """Gives the size of the subtree of this node, in number of nodes. Since
        this is a terminal node, is always 1."""
        return 1


class Individual:
    """Represents a GP individual"""

    def __init__(self, program):
        self.program = program
        self.errors = None
        self.total_error = None

    def __str__(self):
        return """Individual with:
  |- Program: {}
  |- Depth: {}
  |- Nodes: {}
  |- Total Error: {}
  |- Errors: {}""".format(self.program, self.program.tree_depth(), self.nodes(),
                          self.total_error, self.errors)

    def evaluate_individual(self, test_cases):
        """Evaluates the individual given a set of test cases. test_cases should
        be a list of (input, output) pairs (i.e. tuples) telling what output
        should be produced given each input. Inputs are themselves dictionaries
        where the variable names are the keys and the values are the variables'
        values, ex: {"x": 5.0, "y", 2.0}. Outputs are floats.

        This function should set the individual's errors (to be a list of errors
        on the test cases) and total_error to be the sum of errors."""

        pass

    def is_solution(self):
        """Returns True if total_error is less than THRESHOLD."""
        return self.total_error < THRESHOLD

    def nodes(self):
        """Number of nodes in the program of this individual."""
        return self.program.size_of_subtree()


def random_terminal():
    """Returns a random terminal node."""

    # Half of the time pick a variable, the other half pick a random
    # float in the range [-10, 10]
    if random.random() < 0.5:
        terminal_value = random.choice(VARIABLES)
    else:
        #terminal_value = random.uniform(-10, 10)
        terminal_value = 1.0

    return TerminalNode(terminal_value)

def initialize_individual():
    """Creates a new individual from scratch using ramped-half-and-half. This means
    you'll need to create full and grow functions. I recommend having max depth
    set in the range [2, 5] inclusive.

    You will want to use the random_terminal function above."""

    pass

def subtree_at_index(node, index):
    """Returns subtree at particular index in this tree. Traverses tree in
    depth-first order, assigning indices in order of traversal."""

    if index == 0:
        return node

    # Subtract 1 for the current node
    index -= 1

    # Go through each child of the node, and find the one that contains this index
    for child in node.children:
        child_size = child.size_of_subtree()
        if index < child_size:
            return subtree_at_index(child, index)
        index -= child_size

    return "INDEX {} OUT OF BOUNDS".format(index)

def replace_subtree_at_index(node, index, new_subtree):
    """Replaces subtree at particular index in this tree. Traverses tree in
    depth-first order, assigning indices in order of traversal."""

    # Return the subtree if we've found index == 0
    if index == 0:
        return new_subtree

    # Subtract 1 for the current node
    index -= 1

    # Go through each child of the node, and find the one that contains this index
    for child_index in range(len(node.children)):
        child_size = node.children[child_index].size_of_subtree()
        if index < child_size:
            new_child = replace_subtree_at_index(node.children[child_index], index, new_subtree)
            node.children[child_index] = new_child
            return node
        index -= child_size

    return "INDEX {} OUT OF BOUNDS".format(index)


def mutation(parent):
    """Mutates an parent (individual) by replacing a random subtree with a randomly
    generated subtree. You should make use of replace_subtree_at_index above, as well
    as the nodes method of Individual.

    Assume that initialize_tree(min, max) (which another group will write) takes
    a minimum depth and a maximum depth, and returns a randomly-generated tree
    with depth in that range. Since you don't have this function yet, you should
    test your code with a hard-coded subtree to add to the parent."""

    pass

def crossover(parent1, parent2):
    """Crosses over two parents (individuals) to create a child program. You
    should make use of subtree_at_index and replace_subtree_at_index above, as well
    as the nodes method of Individual."""

    pass


def tournament_selection(population, tournament_size):
    """Selects an individual from the population using tournament selection
    with given tournament size."""

    pass

def make_test_cases():
    """Makes a list of test cases. Each test case is a tuple where the first
    element is a dictionary containing the x and y assignments, and the second
    element is the correct output.

    You should pick a function f(x, y) = something, and hard-code the correct
    outputs for that function. You should have somewhere between 50 and 200 test
    cases."""

    pass



def main():

    # This program represents (+ (* x 5.0) y)
    prog1 = FunctionNode("+",
                [FunctionNode("*",
                   [TerminalNode("x"),
                    TerminalNode(5.0)]),
                 TerminalNode("y")])

    # This program represents (- (sin (/ 1.0 2.0)) (exp y))
    prog2 = FunctionNode("-",
                [FunctionNode("sin",
                    [FunctionNode("/",
                        [TerminalNode(1.0),
                         TerminalNode(2.0)])
                    ]),
                 FunctionNode("exp",
                    [TerminalNode("y")])
                    ])

    print("prog1:", prog1)
    print("Depth:", prog1.tree_depth())
    print()

    assignments = {"x": 7.0, "y": 9.0}

    print("prog1({}) =".format(assignments), prog1.eval(assignments))
    print("prog2({}) =".format(assignments), prog2.eval(assignments))

    print("\n--------- Making individuals ----------")

    ind1 = Individual(prog1)
    print(ind1)
    print()

    ind2 = Individual(prog2)
    print(ind2)
    print()



if __name__ == "__main__":
    main()
