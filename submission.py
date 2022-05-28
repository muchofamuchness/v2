from Agent import Agent, AgentGreedy
from TaxiEnv import TaxiEnv, manhattan_distance
import random
import math
import multiprocessing
import time
import ctypes

#time_return_limit = 100
#exit_time = 1

op_to_int = {'move north' : 0, 'move south':1, 'move east':2, 'move west':3, 'refuel':4, 'pick up passenger':5, 'drop off passenger':6, 'park':7}

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

    def minimax(self, env, agent, turn, depth):
        if env.done() or depth == 0:
            return self.heuristic(env, agent)

        operators = env.get_legal_operators(turn)

        children = [self.create_child(env.clone(), turn, op) for op in operators]
        children_heuristics = [(self.return_value(self.minimax(child, agent, 1 - turn, depth - 1)), op) for
                              child, op in zip(children, operators)]

        if turn == agent:
            return max(children_heuristics, key=lambda x: x[0])

        else:
            return min(children_heuristics, key=lambda x: x[0])

    def process_run(self, env, agent_id, operator):
        depth = 1
        while True:
            result = self.minimax(env, agent_id, agent_id, depth)[1]
            operator.value = env.get_legal_operators(agent_id).index(result)
            
            depth += 1

    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        operator = multiprocessing.Value('i', 0)
        start = time.time()
        proc = multiprocessing.Process(target=self.process_run, args=(env, agent_id, operator))
        proc.start()
        proc.join(time_limit * 0.9)
        proc.terminate()
        # print(f"Run time: {time.time() - start}")
        return env.get_legal_operators(agent_id)[operator.value]


class AgentAlphaBeta(Agent):
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
        if isinstance(a, int) or isinstance(a, float):
            return a
        return a[0]

    def create_child(self, child, taxi_id, op):
        child.apply_operator(taxi_id, op)
        return child



    def alphabeta(self, env, agent, depth, alpha, beta):

        operators = env.get_legal_operators(agent)
        children = [self.create_child(env.clone(), agent, op) for op in operators]

        curr_max = -math.inf
        curr_max_op = None

        for child, op in zip(children, operators):
            v = self._alphabeta_(child, agent, 1-agent, depth - 1, alpha, beta)
            if v > curr_max:
                curr_max_op = op
                curr_max = v

        return curr_max_op, curr_max

    def _alphabeta_(self, env, agent, turn, depth, alpha, beta):
        if env.done() or depth == 0:
            return self.heuristic(env, agent)

        operators = env.get_legal_operators(turn)
        children = [self.create_child(env.clone(), turn, op) for op in operators]

        if turn == agent:
            curr_max = -math.inf

            for child, op in zip(children, operators):
                child_heuristics = self.return_value(self.alphabeta(child, agent, 1-turn, depth - 1, alpha, beta))
                # v = (child_heuristics, op)
                if child_heuristics > current_max[0]:
                    current_max = (child_heuristics, op)
                alpha = max(current_max[0], alpha)
                if current_max[0] >= beta:
                    return (math.inf, op)
            return current_max

            for child, op in zip(children, operators):
                child_heuristics = self.return_value(self.alphabeta(child, agent, 1-turn, depth - 1, alpha, beta))
                # v = (child_heuristics, op)
                if child_heuristics < current_min[0]:
                    current_min = (child_heuristics, op)
                #current_min = min([child_heuristics, current_min], key=lambda x: x[0])
                beta = min(current_min[0], beta)
                if current_min[0] <= alpha:
                    return (-math.inf, op)
            return current_min

    def process_run(self, env, agent_id, operator):
        depth = 1

        curr_max_h = -math.inf
        while True:
            result, h = self.alphabeta(env, agent_id, depth, -math.inf, math.inf)

            if curr_max_h < h:
                curr_max_h = h
                operator.value = env.get_legal_operators(agent_id).index(result)

            depth += 1

    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        operator = multiprocessing.Value('i', 0)
        start = time.time()
        proc = multiprocessing.Process(target=self.process_run, args=(env, agent_id, operator))
        proc.start()
        proc.join(time_limit * 0.85)
        proc.terminate()
        # print(f"Run time: {time.time() - start}")
        return env.get_legal_operators(agent_id)[operator.value]


class AgentAlphaBeta1(Agent):
    def __init__(self):
        self.alpha = -math.inf
        self.beta = math.inf

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
        if isinstance(a, int) or isinstance(a, float):
            return a
        return a[0]

    def create_child(self, child, taxi_id, op):
        child.apply_operator(taxi_id, op)
        return child



    def alphabeta(self, env, agent, depth, alpha, beta):

        operators = env.get_legal_operators(agent)
        children = [self.create_child(env.clone(), agent, op) for op in operators]

        curr_max = -math.inf
        curr_max_op = None

        for child, op in zip(children, operators):
            v = self._alphabeta_(child, agent, 1-agent, depth - 1, alpha, beta)
            if v > curr_max:
                curr_max_op = op
                curr_max = v

        return curr_max_op, curr_max

    def _alphabeta_(self, env, agent, turn, depth, alpha, beta):
        if env.done() or depth == 0:
            return self.heuristic(env, agent)

        operators = env.get_legal_operators(turn)
        children = [self.create_child(env.clone(), turn, op) for op in operators]

        if turn == agent:
            curr_max = -math.inf

            for child, op in zip(children, operators):
                v = self._alphabeta_(child, agent, 1-turn, depth - 1, alpha, beta)
                curr_max = max(curr_max, v)

                alpha = max(alpha, curr_max)
                if curr_max >= beta:
                    return math.inf

            return curr_max

        else:
            curr_min = math.inf

            for child, op in zip(children, operators):
                v = self._alphabeta_(child, agent, 1-turn, depth - 1, alpha, beta)

                curr_min = min(v, curr_min)

                beta = min(beta, curr_min)

                if curr_min <= alpha:
                    return -math.inf

            return curr_min

    def process_run(self, env, agent_id, operator):
        depth = 1

        curr_max_h = -math.inf
        while True:
            result, h = self.alphabeta(env, agent_id, depth, -math.inf, math.inf)

            if curr_max_h < h:
                curr_max_h = h
                operator.value = env.get_legal_operators(agent_id).index(result)

            depth += 1

    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        operator = multiprocessing.Value('i', 0)
        start = time.time()
        proc = multiprocessing.Process(target=self.process_run, args=(env, agent_id, operator))
        proc.start()
        proc.join(time_limit * 0.9)
        proc.terminate()
        # print(f"Run time: {time.time() - start}")
        return env.get_legal_operators(agent_id)[operator.value]



class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()
