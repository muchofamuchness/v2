import time

from Agent import Agent, AgentGreedy
import math
from TaxiEnv import TaxiEnv, manhattan_distance
import random

class AgentGreedyImproved(AgentGreedy):
    # TODO: section a : 3
    def run_step(self, env: TaxiEnv, taxi_id, time_limit):
        operators = env.get_legal_operators(taxi_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(taxi_id, op)
        children_heuristics = [self.heuristic(child, taxi_id) for child in children]
        max_heuristic = max(children_heuristics)
        index_selected = children_heuristics.index(max_heuristic)
        return operators[index_selected]

    # start with 16 fuel

    def heuristic(self, env: TaxiEnv, taxi_id: int):
        reward = 0
        taxi = env.get_taxi(taxi_id)
        other_taxi = env.get_taxi((taxi_id + 1) % 2)

        reward += (taxi.cash - other_taxi.cash) * 10 # more / less reward?
        reward += taxi.fuel * 5
        closest_station_id = 0 if (manhattan_distance(env.gas_stations[0].position, taxi.position) <=
                                     manhattan_distance(env.gas_stations[1].position, taxi.position)) else 1
        destination = None
        if len([(manhattan_distance(passenger.position, taxi.position), i) for i, passenger in
                enumerate(env.passengers)]) == 0:
            destination = taxi.passenger.destination

        else:
            closest_passenger = min([(manhattan_distance(passenger.position, taxi.position), i) for i, passenger in
                                     enumerate(env.passengers)], key=lambda x: x[0])

            destination = taxi.passenger.destination if (env.taxi_is_occupied(taxi_id)) \
                else env.passengers[closest_passenger[1]].position

        if (taxi.fuel < manhattan_distance(taxi.position, destination) +
                        min([manhattan_distance(destination, station.position) for station in env.gas_stations])):
            reward += 8 - manhattan_distance(taxi.position, env.gas_stations[closest_station_id].position)
        else:
            reward += 8 - manhattan_distance(taxi.position, destination)
        return reward


class AgentMinimax(Agent):

    stop_thread = False

    def __init__(self):
        self.best_op = 0
        self.stop_thread = False


    # def heuristic(self, env: TaxiEnv, agent: int, turn: int):
    def heuristic(self, env: TaxiEnv, taxi_id: int):
        reward = 0

        taxi = env.get_taxi(taxi_id)
        other_taxi = env.get_taxi((taxi_id + 1) % 2)

        reward += (taxi.cash - other_taxi.cash) * 10  # more / less reward?
        reward += taxi.fuel * 5
        closest_station_id = 0 if (manhattan_distance(env.gas_stations[0].position, taxi.position) <=
                                   manhattan_distance(env.gas_stations[1].position, taxi.position)) else 1

        destination = None
        if len([(manhattan_distance(passenger.position, taxi.position), i) for i, passenger in
                enumerate(env.passengers)]) == 0:
            destination = taxi.passenger.destination

        else:
            closest_passenger = min([(manhattan_distance(passenger.position, taxi.position), i) for i, passenger in
                                     enumerate(env.passengers)], key=lambda x: x[0])

            destination = taxi.passenger.destination if (env.taxi_is_occupied(taxi_id)) \
                else env.passengers[closest_passenger[1]].position

        if (taxi.fuel < manhattan_distance(taxi.position, destination) +
                min([manhattan_distance(destination, station.position) for station in env.gas_stations])):
            reward += 8 - manhattan_distance(taxi.position, env.gas_stations[closest_station_id].position)
        else:
            reward += 8 - manhattan_distance(taxi.position, destination)
        return reward

    def return_value(self, a):
        if isinstance(a, int):
            return a
        return a[0]

    def create_child(self, child, taxi_id, op):
        child.apply_operator(taxi_id, op)
        return child

    def minmax(self, env, agent, turn, depth, stop):

        if env.done() or depth == 0:
            return self.heuristic(env, agent)

        operators = env.get_legal_operators(turn)
        children = [self.create_child(env.clone(), turn, op) for op in operators]

        children_heuristics = [(self.return_value(self.minmax(child, agent, 1-turn, depth - 1, stop)), op) for child, op in zip(children, operators)]

        if turn == agent:
            return max(children_heuristics, key=lambda x: x[0])

        else:
            return min(children_heuristics, key=lambda x: x[0])

    def worker(self, env, agent, stop, best_opp):
        depth = 1

        while not stop():
            re = self.minmax(env, agent, agent, depth, stop)
            depth += 1

            if isinstance(re, int):
                best_opp[0] = env.get_legal_operators(agent)[0]

            best_opp[0] = re[1]



    # TODO: section b : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):

        # re = self.minmax(env, agent_id, agent_id, 7, lambda : False)
        # #
        # if isinstance(re, int):
        #     return env.get_legal_operators(agent_id)[0]
        # #
        # return re[1]
        #

        # Python program killing
        # a thread using multiprocessing
        # module

        # import multiprocessing
        #
        #
        # process = multiprocessing.Process(target=self.worker, args=(env, agent_id, lambda: False))
        # process.start()
        #
        # time.sleep(time_limit * 0.97)
        # process.terminate()
        #
        # return self.best_op
        from threading import Thread
        # import time
        #
        best_opp = [None]
        #
        thread = Thread(target=self.worker, args=(env, agent_id, lambda: self.stop_thread, best_opp))
        thread.start()
        thread.join(time_limit * 0.7)
        print("exit")
        return self.best_op
        #
        # time.sleep(time_limit * 0.9)
        #
        # self.stop_thread = True
        # # thread.join()
        #
        # return best_opp[0]

        # while not self.stop_thread:
        #
        #     thread = Thread(target=self.thread_run, args=(env, agent_id, depth, lambda : self.stop_thread))
        #     thread.start()
        #
        #     print(depth)
        #     while thread.is_alive():
        #         curr_time = time.time()
        #
        #         if (curr_time - start_time) > time_limit * 0.93:
        #             self.stop_thread = True
        #             thread.join()
        #
        #             return self.best_op
        #
        #
        #
        #     depth += 1
        # return self.best_op
        # return re[1]


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()
