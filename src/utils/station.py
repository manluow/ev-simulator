import numpy as np
import pandas as pd
import utm


class Station(object):

    def __init__(self, shop_id, shop_name, latitude, longitude, park_num, car_list, hex_dict, tree):
        self._shop_id = shop_id
        self._shop_name = shop_name
        self._latitude = latitude
        self._longitude = longitude
        self._park_num = park_num
        
        # Transfer the its original coordinate to utm coordinate
        point = utm.from_latlon(latitude, longitude)[0:2]
        hex_utm = tree.search_knn(point, 1)[0][0].data
        self._hex_id = hex_dict[hex_utm]
        if self._park_num < len(car_list):
            print(self._park_num)
            print(len(car_list))
            raise ValueError('Too many cars for limited parking space!')
        self._car_list = []
        for car in car_list:
            self._car_list.append(car)

    def get_shop_id(self):
        return self._shop_id

    def get_shop_name(self):
        return self._shop_name

    def get_location(self):
        return (self._latitude, self._longitude)

    def get_hex_id(self):
        # Each car shall be assigned to one single hexagon grid
        return self._hex_id

    def get_remain_park_num(self):
        # Calculate the parking space remained
        return self._park_num - len(self._car_list)

    def get_available_car(self):
        return self._car_list

    def accept_order(self, order_id, destination, order_lasting_time, car_dict):
        if len(self._car_list) > 0:
            best_one = self._car_list[0]
            best_car = car_dict[best_one]
            for i in self._car_list:
                car = car_dict[i]  # get the instance of the very car
                if best_car.get_running_time() <= car.get_running_time():
                    best_car = car
                    best_one = i

            running_time = best_car.get_running_time()
            if running_time >= order_lasting_time * 1.5:
                # The buffer size of battery quantity is 1.5x
                # e.g. Even a car's battery quantity can last 10km, the user who wishes to drive for 10km would not choose it.
                # Instead, the user would choose a car that can run at least 15km.
                best_car.assign_order(order_id, destination, order_lasting_time)
                self.use_one_car(best_one)
                return best_one
            if best_car._running_ability < order_lasting_time * 1.5:
                return -3
            return -1
        else:
            # Not enough cars available!
            return -2

    def use_one_car(self, car_id):
        # If one car has been chosen, then it shall be removed from the car list of the station.
        if self._car_list != []:
            self._car_list.remove(car_id)

    def add_one_car(self, car_id):
        # Add one car to this station (Add a new car or park an existing car).
        if car_id not in self._car_list and self._park_num >= len(self._car_list) + 1:
            self._car_list.append(car_id)
            return True
        else:
            return False

    def check_avalibility(self, car_id):
        # Check whether there is enough parking space for a car wishing to park here.
        if car_id not in self._car_list and self._park_num >= len(self._car_list) + 1:
            return True
        else:
            return False


if __name__ == '__main__':
    path = '../influx/station_operation.csv'
    station_data = pd.read_csv(path)

    station_data_unique = []
    for line in station_data.values:
        shop_id = line[0]
        shop_name = line[1]
        latitude = line[2]
        longitude = line[3]
        park_num = line[4]
        temp = [shop_id, shop_name, latitude, longitude, park_num]
        if temp not in station_data_unique:
            station_data_unique.append(temp)
    print(len(station_data_unique))

    car_id = 0
    station_list = []
    for line in station_data_unique:
        shop_id = line[0]
        shop_name = line[1]
        latitude = line[2]
        longitude = line[3]
        park_num = line[4]

        if park_num != 0:
            station = Station(shop_id, shop_name, latitude, longitude, park_num, [car_id])
            car_id += 1
            station_list.append(station)
        else:
            station = Station(shop_id, shop_name, latitude, longitude, park_num, [])
            station_list.append(station)

    for station in station_list:
        print(station.get_available_car())
