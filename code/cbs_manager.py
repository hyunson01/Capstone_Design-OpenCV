from cbs_basic import CBSSolver
from icbs_cardinal_bypass import ICBS_CB_Solver
from icbs_complete import ICBS_Solver
from visualize import Animation
from single_agent_planner import get_sum_of_cost
from path_relay import set_paths


class CBSManager:
    def __init__(self, solver_type="ICBS", disjoint=False, visualize_result=True):
        self.solver_type = solver_type
        self.disjoint = disjoint
        self.visualize_result = visualize_result

    def load_instance(self, my_map, starts, goals, ids):
        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.ids = ids

    def create_solver(self):
        if self.solver_type == "CBS":
            return CBSSolver(self.my_map, self.starts, self.goals)
        elif self.solver_type == "ICBS_CB":
            return ICBS_CB_Solver(self.my_map, self.starts, self.goals)
        elif self.solver_type == "ICBS":
            return ICBS_Solver(self.my_map, self.starts, self.goals)
        else:
            raise ValueError(f"Unknown solver type: {self.solver_type}")

    def run(self):
        solver = self.create_solver()
        result = solver.find_solution(self.disjoint)

        if result is None:
            print("No solution found.")
            return None

        paths, nodes_generated, nodes_expanded = result
        cost = get_sum_of_cost(paths)

        set_paths(self.ids, self.starts, paths)

        print(f"Total cost: {cost}")
        print(f"Nodes generated: {nodes_generated}, Nodes expanded: {nodes_expanded}")
        if self.visualize_result:
            animation = Animation(self.my_map, self.starts, self.goals, paths)
            animation.show()

        return paths
