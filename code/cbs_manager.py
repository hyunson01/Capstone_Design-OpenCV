<<<<<<< HEAD
=======
import sys
import os

# D:\git\MAPF-ICBS\code 경로를 추가
sys.path.append(r"D:\git\MAPF-ICBS\code")


>>>>>>> 2a2ebcd35adb08fa2b43280641b59c6f47880fb5
from cbs_basic import CBSSolver
from icbs_cardinal_bypass import ICBS_CB_Solver
from icbs_complete import ICBS_Solver
from visualize import Animation
from single_agent_planner import get_sum_of_cost


class CBSManager:
<<<<<<< HEAD
    def __init__(self, solver_type="ICBS", disjoint=False, visualize_result=True):
=======
    def __init__(self, solver_type="CBS", disjoint=False, visualize_result=True):
>>>>>>> 2a2ebcd35adb08fa2b43280641b59c6f47880fb5
        self.solver_type = solver_type
        self.disjoint = disjoint
        self.visualize_result = visualize_result

    def load_instance(self, my_map, starts, goals):
        self.my_map = my_map
        self.starts = starts
        self.goals = goals

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
        print(f"Total cost: {cost}")
        print(f"Nodes generated: {nodes_generated}, Nodes expanded: {nodes_expanded}")

        if self.visualize_result:
            animation = Animation(self.my_map, self.starts, self.goals, paths)
            animation.show()

        return paths
