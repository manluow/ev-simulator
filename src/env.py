from datetime import datetime, date, timedelta
import collections
import copy
import math
import random
import time
import utm
import pandas as pd
import kdtree
import numpy as np
import pickle
from pathlib import Path
import prettytable as pt


from src.utils.order import Order
from src.utils.station import Station
from src.utils.car import Car

class Env(object):

    def __init__(self, s_time=None, station_count=2100, car_count=5000):
        self.training = False
        self._offline_speed = 1.0
        self.name_log_file()
        # initialization of time-related parameters
        self._init_time = s_time if s_time else datetime(2020, 4, 1, 0, 0, 0)
        self._current_time = self._init_time
        self._running_time = 0
        self._policy_name = 'No rebalancing'

        # initialization of order-related parameters
        self._total_income = 0
        self._red_pack = 0
        self._finished_orders = collections.OrderedDict()  # finished orders
        self._running_orders = collections.OrderedDict()  # running orders
        self._new_orders = collections.OrderedDict()  # new orders
        self._unfulfilled_orders = collections.OrderedDict()  # unfilled orders
        self._lack_electricity = 0  # failed orders for cars lacking battery quantity
        self._lack_car = 0  # failed orders for cars not enough
        self._exceed_limitation = 0  # the number of cars which can not handle orders
        self._fake_unfulfilled_orders = collections.OrderedDict()
        self._to_be_parked_orders = []  # ids of finishing orders

        self._order_generator = None

        # initialization of car-related parameters
        self._no_dest_car = 0  # the number of cars which can not find their destinations
        self._car_dict = {}  # the dictionary of cars
        self._cars_rd = []  # running cars
        self._cars_p = []  # stopped cars
        self._car_count = car_count
        self._no_parking_car = 0
        self._new_car = 0  # new cars
        self._unit_price = 0.5  # the unit price of cars: 0.5
        self._rebalance_time = 0  # rebalancing times
        self.init_cars()

        # initialization of station-related parameters
        self._hex_dict = {}
        self._hex_reward = {}
        self._hex_rebalanced_car = {}
        self._hex_index_dict = {}
        self._hex_list = []
        self._global_order_amount = 0
        self._hex_info_dict = {}
        self._hex_tree = None
        self._station_dict = {}
        self._position_station_dict = {}
        self._station_position_dict = {}
        self._fake_station = {}
        self._candidate_stations = self.get_candidate_stations()
        self._current_stations = None
        self._offline_stations = []
        self._station_count = station_count
        self._station_positions = []
        self._kdtree = None
        self._order_station_count = collections.OrderedDict()
        self._station_data_unique = {}
        self._online_car_ratio = 0
        self._average_distance_dict = {}
        self.init_stations()

        # station-online-related and station-offline-related parameters
        self._offline_loop = 1440 * 7
        self._online_loop = 1440 * 7
        self._offline_stations_n_per_loop = self.get_close_station_num()
        self._online_stations_n_per_loop = self.get_open_station_num()
        self._offline_stations_n_per_day = 0  # calculated in every period
        self._online_stations_n_per_day = 0
        self._stations_to_be_removed = []
        self._stations_to_be_added = []
    
        with open('data/daily_online_offline.pkl', 'rb') as f:
            self._offline_online_seq  = pickle.load(f)
            f.close()
        self._daily_online_seq = self._offline_online_seq['online_seq']
        self._daily_offline_seq = self._offline_online_seq['offline_seq']

        np.random.seed(12)

        # statistic data
        self._tl = []
        self._cars_rd_tl = []
        self._cars_p_tl = []
        self._cars_no_parking_tl = []
        self._no_dest_car_tl = []
        self._online_cars_tl = []
        self._running_orders_tl = []
        self._new_orders_tl = []
        self._unfulfilled_orders_tl = []
        self._unfulfilled_orders_total_tl = []
        self._no_dest_car_tl = []


        # single step-related
        self._cars_to_be_updated = {}
        hex_pos_info = pd.read_csv('./data/hex_pos_info.csv')
        self._valid_hexs = hex_pos_info[(hex_pos_info['Valid']==1)]['hex'].tolist()

    def name_log_file(self):
        self._log_file = 'logs/log.txt'
        self.file = open(self._log_file, 'w')
        self.file.close()

    def init_cars(self):
        for i in range(0, self._car_count):
            car_id = i
            car = Car(car_id, 100)
            self._car_dict[car_id] = car
            self._cars_p.append(car._car_id)

    def init_stations(self):
        station_info = pd.read_csv('./data/shop_seq_info.csv')
        self._current_stations = station_info[(station_info['Online']==1)]['SHOP_SEQ'].tolist()
        self._offline_stations = station_info[(station_info['Online']==0)]['SHOP_SEQ'].tolist()
        station_data = station_info[['SHOP_SEQ', 'PARK_NUM']]
        hex_pos_info = pd.read_csv('./data/hex_pos_info.csv')
        hex_data = hex_pos_info[['lat', 'long']]
        park_sum = 0

        hex_index = 0
        for point_gps in hex_data.values:
            point_utm = utm.from_latlon(point_gps[0], point_gps[1])[0:2]
            self._hex_list.append(point_utm)
            self._hex_dict[point_utm] = hex_index
            self._hex_index_dict[hex_index] = point_utm
            self._hex_reward[hex_index] = 0
            self._global_order_amount = 0
            self._hex_info_dict[hex_index] = {
                'order_total': 0,
                'order_unfilled': 0,
                'total_time': 0,
                'available_car': 0,
                'coming_car': 0
            }
            self._hex_rebalanced_car[hex_index] = 0
            hex_index += 1
        self._hex_tree = kdtree.create(self._hex_list)

        for line in station_data.values:
            shop_id = line[0]
            shop_name = 'station' + str(shop_id)
            # not a true gps point
            latitude = 30
            longitude = 130
            park_num = int(line[1])
            temp = [shop_id, shop_name, latitude, longitude, park_num]
            self._position_station_dict[(latitude, longitude)] = shop_id
            self._station_position_dict[shop_id] = (latitude, longitude)
            if shop_id not in self._station_data_unique:
                self._station_data_unique[shop_id] = temp

        for k, d in self._station_data_unique.items():
            shop_id = d[0]
            shop_name = 'station' + str(shop_id)
            # not a true point
            latitude = 30
            longitude = 130
            park_num = int(d[4])
            station = Station(shop_id, shop_name, latitude, longitude, park_num, [], self._hex_dict, self._hex_tree)
            if shop_id in self._current_stations:
                self._station_positions.append([latitude, longitude])
            self._station_dict[shop_id] = station
            park_num = station.get_remain_park_num()
            park_sum += park_num
        self._online_car_ratio = self._car_count / park_sum
        self._kdtree = kdtree.create(self._station_positions)

        car_id = 0
        for k, station in self._station_dict.items():
            park_num = station.get_remain_park_num()
            car_num = int((park_num / park_sum) * self._car_count)
            for index in range(car_id, car_id + car_num):
                if not station.add_one_car(index):
                    raise ValueError('Car Assign Error')
                else:
                    car = self._car_dict[index]
                    car.dispatch_to_station(k)
            car_id += car_num

            if k in self._current_stations:
                self._order_station_count[k] = {
                    'count': 0,
                    'orders': []
                }

        for index in range(car_id, self._car_count):
            for _, station in self._station_dict.items():
                if station.add_one_car(index):
                    car = self._car_dict[index]
                    car.dispatch_to_station(station.get_shop_id())
                    break

    
    def get_candidate_stations(self):
        # Get all existing stations
        station_info = pd.read_csv('./data/shop_seq_info.csv')
        shop_info_index = station_info['SHOP_SEQ']
        self.shop_seq = pd.DataFrame({'SHOP_SEQ': shop_info_index.tolist()})
        self.candidate_station = station_info[(station_info['Valid']==1)]['SHOP_SEQ'].tolist()
        return self.candidate_station


    def calculate_current_order_count(self):
        # Get the total number of current orders
        station_dict = {}
        current_t = (self._current_time + timedelta(minutes=10)).strftime('%Y%m%d%H%M')

        for current in [current_t]:

            order_current = self._new_orders[current]
            for k, d in order_current.items():
                if d.get_begin_st() not in self._current_stations:
                    continue
                if d.get_begin_st() not in station_dict:
                    station_dict[d.get_begin_st()] = 1
                else:
                    station_dict[d.get_begin_st()] += 1

        self._current_order_count = station_dict


    def claculate_station_unfilled_order(self, hex_id):
        # Get failed orders
        current_t = (self._current_time + timedelta(minutes=10)).strftime('%Y%m%d%H%M')

        total = 0
        for current in [current_t]:
            order_current = self._new_orders[current]
            for k, d in order_current.items():
                start_id = d.get_begin_st()
                if self._station_dict[start_id].get_hex_id() == hex_id:
                    if k in self._fake_unfulfilled_orders.keys():
                        total += d.get_cost_time() * 0.5
                    else:
                        total += d.get_cost_time() * 0.25

        return total


    def update_global_status(self):
        # Update global status
        station_list = {}
        order_list = {}
        self._global_order_amount = 0
        for hex_id in self._valid_hexs:
            self._hex_info_dict[hex_id] = {
                'order_total': 0,
                'order_unfilled': 0,
                'total_time': 0,
                'available_car': 0,
                'coming_car': 0
            }
            order_list[hex_id] = []

        current_t = (self._current_time + timedelta(minutes=10)).strftime('%Y%m%d%H%M')
        for k in self._current_stations:
            d = self._station_dict[k]
            hex_id = d.get_hex_id()
            if hex_id in self._valid_hexs:
                self._hex_info_dict[hex_id]['available_car'] += len(d.get_available_car())

        for current in [current_t]:
            order_current = self._new_orders[current]
            for k, d in order_current.items():
                begin_shop_id = d.get_begin_st()
                station = self._station_dict[begin_shop_id]
                hex_id = station.get_hex_id()
                if hex_id in self._valid_hexs:
                    order_list[hex_id].append(d)

        for key in self._running_orders:
            order = self._running_orders[key]
            end_shop_id = order.get_end_st()
            station = self._station_dict[end_shop_id]
            hex_id = station.get_hex_id()
            if hex_id in self._valid_hexs:
                end_time = order.get_begin_time() + timedelta(minutes=order.get_cost_time())
                if end_time >= self._current_time + timedelta(minutes=10) and end_time < self._current_time + timedelta(
                        minutes=30):
                    self._hex_info_dict[hex_id]['available_car'] += 1
                    self._hex_info_dict[hex_id]['coming_car'] += 1

        for order in order_list:
            for o in order_list[order]:
                self._global_order_amount += 1
                begin_shop_id = o.get_begin_st()
                station = self._station_dict[begin_shop_id]
                hex_id = station.get_hex_id()
                if hex_id in self._valid_hexs:
                    self._hex_info_dict[hex_id]['order_total'] += 1
                    self._hex_info_dict[hex_id]['total_time'] += o.get_cost_time()
                    order_id = o.get_order_id()
                    if order_id in self._fake_unfulfilled_orders.keys():
                        self._hex_info_dict[hex_id]['order_unfilled'] += 1


    def get_close_station_num(self):
        # Get the number of stations shall be closed in one week
        average_count = int(10 * self._offline_speed)
        return average_count + np.random.randint(-4, 4)


    def get_open_station_num(self):
        # Get the number of stations shall be open in one week
        average_count = int(45 * self._offline_speed)
        return average_count + np.random.randint(-10, 10)


    def move_closed_staion_car(self, shop_id):
        # Remove all the cars in stations to be closed
        shop_pos = self._station_position_dict[shop_id]
        closed_station = self._station_dict[shop_id]
        car_list = closed_station._car_list
        knn_points = self._kdtree.search_knn(shop_pos, 20)

        for point in knn_points:
            if (point[0].data[0], point[0].data[1]) == shop_pos:
                continue

            station_id = self._position_station_dict[(point[0].data[0], point[0].data[1])]
            station = self._station_dict[station_id]

            while station.get_remain_park_num() > 0:
                if len(car_list) == 0:
                    break
                station._car_list.append(car_list[0])
                car_list.pop(0)

            if len(car_list) == 0:
                break


    def wrap_orders(self, orders):
        # Generate new orders
        count = 1
        current_t = self._current_time.strftime('%Y%m%d%H%M')
        self._new_orders[current_t] = collections.OrderedDict()
        for k, d in orders.items():  # start station
            for order in d:  # each order in the very station
                order_id = self._current_time.strftime('%Y%m%d%H%M') + str(count).zfill(3)
                self._new_orders[current_t][order_id] = \
                    Order(order_id=order_id,
                          car_id=None,
                          begin_st=k,
                          end_st=order['destination'],
                          begin_time=self._current_time,
                          cost_time=order['cost_time'])
                count += 1


    def wrap_orders_all_day(self, orders, c_time):
        # Generate new orders of one single day
        count = 1
        current_t = c_time.strftime('%Y%m%d%H%M')
        self._new_orders[current_t] = collections.OrderedDict()
        for k, d in orders.items():  # start station
            for order in d:  # each order in the very station
                order_id = c_time.strftime('%Y%m%d%H%M') + str(count).zfill(3)
                self._new_orders[current_t][order_id] = \
                    Order(order_id=order_id,
                          car_id=None,
                          begin_st=k,
                          end_st=order['destination'],
                          begin_time=c_time,
                          cost_time=order['cost_time'])
                count += 1


    def process_orders(self):
        # Process orders
        current_t = self._current_time.strftime('%Y%m%d%H%M')
        c_orders = self._new_orders[current_t]

        # Process new orders
        for k, d in c_orders.items():
            if d._begin_st not in self._current_stations:
                continue
            result = self._station_dict[d._begin_st].accept_order(
                order_id=d._order_id, destination=d._end_st, order_lasting_time=d._cost_time, car_dict=self._car_dict)

            if result == -1:
                self._lack_electricity += 1
            if result == -2:
                self._lack_car += 1
            if result == -3:
                self._exceed_limitation += 1
            if result >= 0:
                d._car_id = result
                self._running_orders[d._order_id] = d
                self._cars_p.remove(self._car_dict[result]._car_id)
                self._cars_rd.append(self._car_dict[result]._car_id)

                if self._station_dict[d._end_st]._hex_id in self._valid_hexs:
                    if d._end_st not in self._cars_to_be_updated:
                        self._cars_to_be_updated[d._end_st] = []
                    self._cars_to_be_updated[d._end_st].append(self._car_dict[result])
            else:
                self._unfulfilled_orders[d._order_id] = d


    def solve_trouble_order(self, trouble_order):
        # When a car failed to park in its original destination station, the functions shall be called
        trouble_car = self._car_dict[trouble_order._car_id]
        end_st_id = trouble_order._end_st
        end_pos = self._station_position_dict[end_st_id]
        knn_points = self._kdtree.search_knn(end_pos, 20)

        for point in knn_points:
            if (point[0].data[0], point[0].data[1]) == end_pos:
                continue
            station_id = self._position_station_dict[(point[0].data[0], point[0].data[1])]
            station = self._station_dict[station_id]
            if station.check_avalibility(trouble_order._car_id):
                trouble_order._end_st = station._shop_id
                trouble_order._cost_time += 10
                trouble_car.change_desination(station_id)
                trouble_car._order_lasting_time += 10
                break


    def update_car_status(self):
        # Update global car status
        for key in self._car_dict:
            car = self._car_dict[key]
            status = car.update()
            # If the car arrives at a station, the id of the station would be returned
            # Or -1 would be returned
            if status != -1:
                station = self._station_dict[status]
                if status not in self._current_stations:
                    trouble_order = self._running_orders[car._order]
                    self.solve_trouble_order(trouble_order)
                    self._no_dest_car += 1
                    continue

                if car._order not in self._to_be_parked_orders:
                    self._to_be_parked_orders.append(car._order)


    def parking_car(self, station, car_id, rebalance=-1):
        # Park a car in a station
        # station: the very station, car_id: the id of the very car
        car = self._car_dict[car_id]
        if station.get_shop_id() not in self._current_stations or not station.add_one_car(car_id):
            trouble_order = self._running_orders[car._order]
            self.solve_trouble_order(trouble_order)
            self._no_parking_car += 1
        else:

            finished_order_id = car.success_parking()
            # car._soc = 100
            self._finished_orders[finished_order_id] = self._running_orders[finished_order_id]

            finished_order = self._finished_orders[finished_order_id]
            if finished_order.get_end_st() != car.get_destination_station():
                finished_order._end_st = car.get_destination_station()
            finished_order._price = finished_order._cost_time * self._unit_price
            self._total_income += finished_order._price
            self._running_orders.pop(finished_order_id)
            self._cars_p.append(car._car_id)
            self._cars_rd.remove(car._car_id)

            if finished_order_id in self._to_be_parked_orders:
                self._to_be_parked_orders.remove(finished_order_id)

            if station.get_shop_id() not in self._order_station_count:
                self._order_station_count[station.get_shop_id()] = {
                    'count': 0,
                    'orders': []
                }
            self._order_station_count[station.get_shop_id()]['count'] += 1
            self._order_station_count[station.get_shop_id()]['orders'].append(self._finished_orders[finished_order_id])

    def daily_offline(self):
        # Get the offline stations of each single day
        stations_to_be_removed = self._daily_offline_seq[self._current_time.strftime('%Y%m%d')]
        for station_id in stations_to_be_removed:
            if station_id in self._current_stations:
                station = self._station_dict[station_id]
                if station_id in self._order_station_count:
                    self._order_station_count.pop(station_id)
                    offline_point = self._station_position_dict[station_id]
                    self._kdtree.remove(offline_point)
                    self.log_info(f'Offline station: {station_id}')
                    self._current_stations.remove(station_id)
                    self.move_closed_staion_car(station_id)
                    self._offline_stations.append(station_id)

        self._station_count = len(self._current_stations)

    def daily_online(self):
        # Get the online stations of each single day
        stations_to_be_added = self._daily_online_seq[self._current_time.strftime('%Y%m%d')]
        for station_id in stations_to_be_added:
            online_point = self._station_position_dict[station_id]
            self._kdtree.add(online_point)
            self._current_stations.append(station_id)
            d = self._station_data_unique[station_id]

            shop_id = d[0]
            shop_name = 'station' + str(shop_id)
            # not a true point
            latitude = 30
            longitude = 130
            park_num = int(d[4])
            station = Station(shop_id, shop_name, latitude, longitude, park_num, [], self._hex_dict, self._hex_tree)
            self.add_new_car(math.ceil(self._online_car_ratio * park_num), station)
            self._station_dict[shop_id] = station

            self.log_info(f'Online station: {shop_id}')

            self._order_station_count[shop_id] = {
                'count': 0,
                'orders': []
            }
            if shop_id in self._offline_stations:
                self._offline_stations.remove(station_id)
        self._station_count = len(self._current_stations)

    def add_new_car(self, number, station):
        # Add new cars in ths simulator and assign them to new stations
        for i in range(self._car_count, self._car_count + number):
            car_id = i
            car = Car(car_id, 100)
            self._car_dict[car_id] = car
            self._cars_p.append(car._car_id)
            if not station.add_one_car(car_id):
                raise ValueError('Added Car Assign Error')
            else:
                car.dispatch_to_station(station.get_shop_id())
                self._new_car += 1
        self._car_count += number
        self.log_info(f'Will add {number} cars to these stations: {station.get_shop_id()}')

    def rebalance_cars_across_hexagon(self, car, action, order_money=-1, average_dis=-1):
        # Explanation: This function is the entrance of rebalancing
        # When the RL model passes action and aiming car to the function, this function would change the destination of the car
        # And statistic data would be updated
        # This function listed here is only for explanation, not for usage

        new_dest_station = action['dest_station']

        if new_dest_station != None:
            new_dest = new_dest_station.get_shop_id()

        if new_dest != -1:
            car.change_desination(new_dest)
            status = car.get_destination_station()
            self._rebalance_time += 1

            latitude_old, longitude_old = old_dest_station.get_location()
            latitude_new, longitude_new = new_dest_station.get_location()
            point1_utm = utm.from_latlon(latitude_old, longitude_old)[0:2]
            point2_utm = utm.from_latlon(latitude_new, longitude_new)[0:2]
            distance = ((point1_utm[0] - point2_utm[0]) ** 2 + (point1_utm[1] - point2_utm[1]) ** 2) ** 0.5 / 1000

            x = 30 * (distance ** 2 / 100)
            self._red_pack += x if x < 20 else 20
            station = self._station_dict[status]

            self.parking_car(station, car._car_id)
            return 0
        else:
            pass

        return -1
    
    
    def snapshot(self, verbose=False):
        # Record and print running information and logs
        if verbose:
            parking_total = 0
            for shop_id in self._current_stations:
                station = self._station_dict[shop_id]
                parking_total += station._park_num


            tb = pt.PrettyTable()
            tb.field_names = [" item ", " amount ", "  item  ", "  amount  "]
            tb.add_row(["Number of stations in operation", len(self._current_stations), "Number of available charging docks", parking_total])
            tb.add_row(["Number of EVs in operation", len(self._car_dict), "Number of EVs in charging", len(self._cars_p)])
            tb.add_row(["Number of newly deployed EVs", self._new_car, "Number of EVs on the road", len(self._cars_rd)])
            tb.add_row(["Number of EVs returned with no available parking", self._no_parking_car, "Number of EVs returned to an offline station", self._no_dest_car])

            tb.add_row(["Number of running orders", len(self._running_orders), "Number of rebalancing operations", self._rebalance_time])
            tb.add_row(["Number of finished orders", len(self._finished_orders), "Total number of unfulfilled orders", len(self._unfulfilled_orders) - self._exceed_limitation])
            tb.add_row(["Number of unfulfilled orders(EV battery low)", self._lack_electricity, "Number of unfulfilled orders(no available EV)", self._lack_car])

            tb.add_row(["Total income (10k)", round(self._total_income/10000, 2), "Total cost on user incentives (10k)", round(self._red_pack/10000, 2)])

            info = tb.get_string()
            info = f'\n\n[Snapshot] {self._current_time} -- {self._policy_name}\n' + info + '\n'

            self.file = open(self._log_file, 'a')
            self.file.write(info)
            self.file.close()

            print(info)

        self._tl.append(self._current_time)
        self._cars_p_tl.append(len(self._cars_p))
        self._cars_rd_tl.append(len(self._cars_rd))
        self._cars_no_parking_tl.append(self._no_parking_car)
        self._new_orders_tl.append(len(self._new_orders[self._current_time.strftime('%Y%m%d%H%M')]))
        self._running_orders_tl.append(len(self._running_orders))
        self._no_dest_car_tl.append(self._no_dest_car)
        if len(self._unfulfilled_orders_total_tl) == 0:
            self._unfulfilled_orders_total_tl.append(len(self._unfulfilled_orders))
            self._unfulfilled_orders_tl.append(len(self._unfulfilled_orders))
        else:
            temp = self._unfulfilled_orders_total_tl[-1]
            self._unfulfilled_orders_total_tl.append(len(self._unfulfilled_orders))
            self._unfulfilled_orders_tl.append(self._unfulfilled_orders_total_tl[-1] - temp)

        self._online_cars_tl.append(self._new_car)


    def step_1(self, eager_mode=False):
        #If there are no finishing orders
        if len(self._to_be_parked_orders) == 0:
            self._current_time = self._init_time + timedelta(minutes=self._running_time)

            # Online and offline operation
            if self._offline_speed != 0:

                if self._running_time % 1440 == 0:
                    self.daily_offline()
                    self.daily_online()

            self.calculate_current_order_count()
            self.process_orders()
            self.update_car_status()

            if self._running_time % 120 == 0:
                self.log_info(f"{self._current_time} Number of generated orders : {len(self._new_orders[self._current_time.strftime('%Y%m%d%H%M')])}")
                self.snapshot(verbose=True)
            else:
                self.snapshot(verbose=False)

            self._running_time += 10
            self.log_info("Current time step:" + str(self._running_time))
            self._no_parking_car = 0
            self._no_dest_car = 0
            self._new_car = 0
            self.update_global_status()

            if len(self._to_be_parked_orders) == 0:
                return []

            to_be_parked_agents = []
            processing_orders = []
            for order_id in self._to_be_parked_orders:
                processing_order = self._running_orders[order_id]
                # Choose different agents
                station = self._station_dict[processing_order._end_st]
                if station._hex_id not in to_be_parked_agents:
                    to_be_parked_agents.append(station._hex_id)
                    processing_orders.append(processing_order)
                    self._to_be_parked_orders.remove(order_id)

            return processing_orders

        else:
            to_be_parked_agents = []
            processing_orders = []
            for order_id in self._to_be_parked_orders:
                processing_order = self._running_orders[order_id]
                # Choose different agents
                station = self._station_dict[processing_order._end_st]
                if station._hex_id not in to_be_parked_agents:
                    to_be_parked_agents.append(station._hex_id)
                    processing_orders.append(processing_order)
                    self._to_be_parked_orders.remove(order_id)

            return processing_orders

    def log_info(self, str):
        print("[info] " + str)
