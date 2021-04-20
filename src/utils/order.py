class Order(object):

    def __init__(self, order_id, car_id, begin_st, end_st, begin_time, cost_time):
        self._order_id = order_id
        self._car_id = car_id
        self._begin_st = begin_st
        self._end_st = end_st
        self._begin_time = begin_time
        self._cost_time = cost_time  # expected cost time
        self._price = self._cost_time * 0.5

        self._is_finished = -1

    def get_car_id(self):
        return self._car_id

    def get_order_id(self):
        return self._order_id

    def get_begin_st(self):
        return self._begin_st

    def get_end_st(self):
        return self._end_st

    def get_begin_time(self):
        return self._begin_time

    def get_cost_time(self):
        return self._cost_time

    def get_order_price(self):
        # Unit price: 0.5
        return self._cost_time * 0.5
