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
    import math

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

    def minmax(self, env, agent, turn, depth):
        if env.done() or depth == 0:
            return self.heuristic(env, turn)

        operators = env.get_legal_operators(turn)
        children = [env.clone() for _ in operators]

        children_heuristics = [(self.minmax(child, agent, 1-turn, depth - 1), op) for child, op in zip(children, operators)]

        if turn == agent:
            return max(children_heuristics, key=lambda x: x[0])

        else:
            return min(children_heuristics, key=lambda x: x[0])

    # TODO: section b : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        
        return self.minmax(env, agent_id, agent_id, 8)[1]


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()
