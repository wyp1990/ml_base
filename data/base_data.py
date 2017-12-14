import abc


class base_data(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def build_input(self, ):
        pass