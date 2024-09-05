class Cartesian:
    def __init__(self) -> None:
        pass

    @property
    def protocol(self, *args, **kwargs):
        raise NotImplementedError
    
    @property
    def format(self, *args, **kwargs):
        raise NotImplementedError
    
    def goto(self, movment_type='Linear', **coordinate):
        raise NotImplementedError

    def open_actuator(self, *args, **kwargs):
        raise NotImplementedError

    def close_actuator(self, *args, **kwargs):
        raise NotImplementedError

    def rotate_actuator(self, *args, **kwargs):
        raise NotImplementedError

    def set_absolute(self, *args, **kwargs):
        raise NotImplementedError

    def set_relative(self, *args, **kwargs):
        raise NotImplementedError


class GcodeCartesian(Cartesian):
    def __init__(self):
        super().__init__()

    def goto(self, movment_type='Linear', **coordinate):
        #! This can be linear, arc or belzier movment. how to differ?
        if movment_type == 'Linear':
            cords = "G0 "
            for axis, pos in coordinate.items():
                cords += f"{axis.upper()}{round(float(pos), 3)} "
            return cords+'\n'
        
    def set_absolute(self, *args, **kwargs):
        return "G90\n"
    
    def set_relative(self, *args, **kwargs):
        return "G91\n"


class CLPCartesian(Cartesian):
    def __init__(self):
        super().__init__()

