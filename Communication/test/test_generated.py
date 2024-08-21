from . import *

#! Genereted Unit Teste |
class TestRegNode(unittest.TestCase):
    def test_reg_node_new(self):
        node_registry = {}
        node = Nodes.reg_node(node_registry, 'A')
        self.assertIsInstance(node, Nodes.Node)
        self.assertEqual(node.name, 'A')
        self.assertEqual(node_registry['A'], node)

    def test_reg_node_existing(self):
        node_registry = {'Nodes.A': Nodes.Node('A')}
        node = Nodes.reg_node(node_registry, 'A')
        self.assertEqual(node, node_registry['A'])

class TestRecipeReader(unittest.TestCase):
    def test_recipe_reader(self):
        node_registry = {}
        callback_registry = {'sum_cal': lambda node: {'0': sum([i.value for i in node.inputs.values() if i.value is not None])}}
        recipe = [
            {
                'id': 'A',
                'callback': 'sum_cal',
                'connections': [
                    {'A:O:0': 'B1:I:0'},
                    {'A:O:0': 'B2:I:0'},
                ]
            },
            {
                'id': 'B1',
                'callback': 'sum_cal',
                'connections': [
                    {'B1:O:0': 'C:I:0'}
                ]
            },
            {
                'id': 'B2',
                'callback': 'sum_cal',
                'connections': [
                    {'B2:O:0': 'C:I:1'}
                ]
            },
            {
                'id': 'C',
                'callback': 'sum_cal',
                'connections': [
                    {'C:O:0': 'A:I:0'}
                ]
            }
        ]

        Nodes.recipe_reader(node_registry, callback_registry, recipe)

        self.assertEqual(len(node_registry), 4)
        self.assertIsInstance(node_registry['A'], Nodes.Node)
        self.assertIsInstance(node_registry['B1'], Nodes.Node)
        self.assertIsInstance(node_registry['B2'], Nodes.Node)
        self.assertIsInstance(node_registry['C'], Nodes.Node)

        self.assertEqual(len(node_registry['A'].next), 2)
        self.assertEqual(len(node_registry['B1'].next), 1)
        self.assertEqual(len(node_registry['B2'].next), 1)
        self.assertEqual(len(node_registry['C'].next), 1)

        self.assertEqual(node_registry['A'].callback, callback_registry['sum_cal'])
        self.assertEqual(node_registry['B1'].callback, callback_registry['sum_cal'])
        self.assertEqual(node_registry['B2'].callback, callback_registry['sum_cal'])
        self.assertEqual(node_registry['C'].callback, callback_registry['sum_cal'])

class TestLink(unittest.TestCase):
    def test_link_value(self):
        link = Nodes.Link(10)
        self.assertEqual(link.value, 10)
        link.value = 20
        self.assertEqual(link.value, 20)

    def test_link_str_repr(self):
        link = Nodes.Link(10)
        self.assertIn(str(link), f"[0x{str(id(link))[-4:]}] 10 ")
        self.assertIn(repr(link), f"[0x{str(id(link))[-4:]}] 10 ")

class TestNode(unittest.TestCase):
    def test_node_init(self):
        node = Nodes.Node('A')
        self.assertEqual(node.name, 'A')
        self.assertEqual(node.outputs, {})
        self.assertEqual(node.inputs, {})
        self.assertEqual(node.next, [])
        self.assertEqual(node.prev, [])
        self.assertEqual(node.callback_executions, 0)
        self.assertIsNone(node.callback)

    def test_node_next_prev(self):
        node_a = Nodes.Node('A')
        node_b = Nodes.Node('B')
        node_c = Nodes.Node('C')

        node_a.next = [node_b, node_c]
        self.assertEqual(node_a.next, [node_b, node_c])
        self.assertEqual(node_b.prev, [node_a])
        self.assertEqual(node_c.prev, [node_a])

    def test_node_attach_input(self):
        node = Nodes.Node('A')
        link = Nodes.Link(10)
        node.attach_input('I0', link)
        self.assertEqual(node.inputs['I0'], link)

    def test_node_reg_output(self):
        node = Nodes.Node('A')
        output_link = node.reg_output('O0')
        self.assertIsInstance(output_link, Nodes.Link)
        self.assertEqual(node.outputs['O0'], output_link)

    def test_node_process(self):
        node = Nodes.Node('A')
        node.callback = MagicMock(return_value={'0': 20})
        node.inputs = {'I0': Nodes.Link(10), 'I1': Nodes.Link(10)}
        node.reg_output('0')
        node.process()
        self.assertEqual(node.output['0'].value, 20)
        self.assertEqual(node.callback_executions, 1)
        node.callback.assert_called_once_with(node)

    def test_node_iter(self):
        node_a = Nodes.Node('A')
        node_b = Nodes.Node('B')
        node_c = Nodes.Node('C')
        node_a.next = [node_b, node_c]

        visited = set()
        for node in node_a:
            visited.add(node)
        self.assertEqual(visited, {node_a, node_b, node_c})


if __name__ == '__main__':
    unittest.main()