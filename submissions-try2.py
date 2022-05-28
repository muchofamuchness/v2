from Agent import Agent, AgentGreedy
import math
from TaxiEnv import TaxiEnv, manhattan_distance
import random

time_return_limit = 100
exit_time = 1


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

    def minmax(self, env, agent, turn, depth, stop, time):
        global exit_time
        start_time = time.time()
        if stop():
            return -1

        if env.done() or depth == 0 or time <= time_return_limit - exit_time * depth:
            return self.heuristic(env, agent)

        operators = env.get_legal_operators(turn)
        children = [self.create_child(env.clone(), turn, op) for op in operators]
        run_time = time.time() - start_time
        children_heuristics = [(self.return_value(self.minmax(child, agent, 1 - turn, depth - 1, stop,
                                                              (run_time / operators.size()))), op)
                               for child, op in zip(children, operators)]

        if turn == agent:
            return max(children_heuristics, key=lambda x: x[0])

        else:
            return min(children_heuristics, key=lambda x: x[0])

    def thread_run(self, env, agent_id, depth, stop):

        if stop():
            return

        re = self.minmax(env, agent_id, agent_id, depth, stop)

        if stop():
            return

        if isinstance(re, int):
            self.best_op = env.get_legal_operators(agent_id)[0]

        self.best_op = re[1]
        return

    # TODO: section b : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):

        # re = self.minmax(env, agent_id, agent_id, 7, lambda : False)
        #
        # if isinstance(re, int):
        #     return env.get_legal_operators(agent_id)[0]
        #
        # return re[1]
        #
        from threading import Thread
        import time

        start_time = time.time()
        self.stop_thread = False
        depth = 1

        while not self.stop_thread:

            thread = Thread(target=self.thread_run, args=(env, agent_id, depth, lambda: self.stop_thread,))
            thread.start()

            print(depth)
            while thread.is_alive():
                curr_time = time.time()
                print((curr_time - start_time))
                if (curr_time - start_time) > time_limit * 0.97:
                    self.stop_thread = True
                    thread.join()
                    return self.best_op

            depth += 1
        return self.best_op
        # return re[1]


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()