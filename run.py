import sys
sys.path.append("../")

import pickle
from datetime import datetime, date, timedelta

from src.env import Env
from src.utils.order import Order

# The initial status of the simulator: 3000 stations and 8000 cars assigned to these stations
simulator = Env(station_count=3000, car_count=8000)
f = open('data/orders_one_week.pkl', 'rb')
simulator._new_orders = pickle.load(f)
f.close()


while simulator._current_time < datetime(2020,4,7, 23, 50, 0):
    # Get the orders going to be finished
    processing_orders = simulator.step_1()
    for processing_order in processing_orders:
        station_id = processing_order.get_end_st()
        station = simulator._station_dict[station_id]
        car = simulator._car_dict[processing_order.get_car_id()]
        simulator.parking_car(station,car._car_id)
