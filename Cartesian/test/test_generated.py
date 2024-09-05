from . import *

class TestCartesian(unittest.TestCase):
    def setUp(self):
        self.cartesian = Cartesian.Cartesian()

    def test_protocol_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.cartesian.protocol

    def test_format_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.cartesian.format

    def test_goto_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.cartesian.goto()

    def test_open_actuator_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.cartesian.open_actuator()

    def test_close_actuator_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.cartesian.close_actuator()

    def test_rotate_actuator_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.cartesian.rotate_actuator()

    def test_set_absolute_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.cartesian.set_absolute()

    def test_set_relative_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.cartesian.set_relative()

class TestGcodeCartesian(unittest.TestCase):
    def setUp(self):
        self.gcode_cartesian = Cartesian.GcodeCartesian()

    def test_goto_linear(self):
        result = self.gcode_cartesian.goto(movment_type='Linear', X=10.1234, Y=20.5678, Z=30.9012)
        self.assertEqual(result, "G0 X10.123 Y20.568 Z30.901 \n")

    def test_goto_default_movement_type(self):
        result = self.gcode_cartesian.goto(X=5, Y=10)
        self.assertEqual(result, "G0 X5.0 Y10.0 \n")

    def test_set_absolute(self):
        result = self.gcode_cartesian.set_absolute()
        self.assertEqual(result, "G90\n")

    def test_set_relative(self):
        result = self.gcode_cartesian.set_relative()
        self.assertEqual(result, "G91\n")

    def test_unimplemented_methods(self):
        with self.assertRaises(NotImplementedError):
            self.gcode_cartesian.protocol
        with self.assertRaises(NotImplementedError):
            self.gcode_cartesian.format
        with self.assertRaises(NotImplementedError):
            self.gcode_cartesian.open_actuator()
        with self.assertRaises(NotImplementedError):
            self.gcode_cartesian.close_actuator()
        with self.assertRaises(NotImplementedError):
            self.gcode_cartesian.rotate_actuator()

class TestCLPCartesian(unittest.TestCase):
    def setUp(self):
        self.clp_cartesian = Cartesian.CLPCartesian()

    def test_unimplemented_methods(self):
        methods = [
            'protocol', 'format', 'goto', 'open_actuator', 'close_actuator',
            'rotate_actuator', 'set_absolute', 'set_relative'
        ]
        for method in methods:
            with self.subTest(method=method):
                with self.assertRaises(NotImplementedError):
                    getattr(self.clp_cartesian, method)()

if __name__ == '__main__':
    unittest.main()
