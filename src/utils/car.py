import numpy as np
import pandas as pd


class Car(object):
    # _soc equals quantity of electric charge
    # [0, 100]

    def __init__(self, car_id, soc):
        self._car_id = car_id
        self._soc = soc
        self._status = 'free'
        self._order = None
        self._current_station = 0
        self._destination_station = 0
        self._run_time = 0
        self._charge_time = 0
        self._running_ability = 150
        self._charging_time = 600

    def get_car_id(self):
        return self._car_id

    def get_soc(self):
        return self._soc
    
    def get_running_time(self):
        # calculate the expected running distance under current battery quantity
        y = self._soc
        lasting_time = (y/100)**2*self._running_ability
        return lasting_time

    def get_destination_station(self):
        return self._destination_station

    def get_current_station(self):
        return self._current_station
    
    def dispatch_to_station(self, station_id):
        # Park this car to one certain station
        self._current_station = station_id
    
    def assign_order(self, order_id, destination, order_lasting_time):
        # Assign one order to this car
        self._status = 'busy'
        self._order = order_id
        self._destination_station = destination
        self._order_lasting_time = order_lasting_time
        
    def change_desination(self, new_destination):
        self._destination_station = new_destination
        # self._order_lasting_time += 10
    
    def update(self):
        # Update the status of this car
        # status dependent: free and running
        if self._status == 'free':
            self._charge_time += 10
            y = self._soc
            lasting_time = (y/100)**2*self._charging_time
            lasting_time += 10
            x = lasting_time
            battery_percentage = x**(1/2)/(self._charging_time**(1/2))*100
            if battery_percentage > 100:
                self._soc = 100
            else:
                self._soc = battery_percentage
        else:
            self._run_time += 10
            y = self._soc
            lasting_time = self._running_ability-(y/100)**2*self._running_ability
            lasting_time += 10
            x = lasting_time
            battery_percentage = ((self._running_ability-x)**(1/2))/(self._running_ability**(1/2))*100
            if type(battery_percentage) != float:
                self._soc = 0
            else:
                self._soc = battery_percentage
            
            if self._run_time >= self._order_lasting_time:                
                return self._destination_station
        return -1
    
    def success_parking(self):
        # When a car finishes its order, it shall be parked to one station and updated
        self._current_station = self._destination_station
        self._status = 'free'
        self._charge_time = 0
        self._run_time = 0
        
        return self._order
                
                
if __name__ == '__main__':
    car = Car(1, 0)
    
    for i in range(0, 100):
        car.update()
        print(car.get_soc())
        if car.get_soc() == 100:
            print('Charge full!')
            break
        
    car.assign_order(1, 1024, 152.3)
    
    count = 1
    for i in range(0, 100):
        car.update()
        print(count, car.get_soc())
        count += 1
        if car.get_soc() == 100:
            print('Charge full!')
            break
