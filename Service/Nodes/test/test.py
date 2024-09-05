from . import *
class TestRecipe(unittest.TestCase):
    def setUp(self):
        def sum_cal(node):
        
            result = sum([i.value for i in node.inputs.values() if i.value is not None])
            
            return {'0': result}
        
        self.sum_cal = sum_cal
        
        self.Nodes = {}
        self.Functions = {'sum_cal': sum_cal}
        self.recipe = [
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


        """
        
               ┌> B1 >┐           
            A >┤      ├> C >┐ 
            │  └> B2 >┘     │
            └<─────<──────<─┘ 
        
        """

    def test_recipe_reader(self):
        Nodes.recipe_reader(self.Nodes, self.Functions, self.recipe)

        # Validate that Nodes are created
        self.assertIn('A', self.Nodes)
        self.assertIn('B1', self.Nodes)
        self.assertIn('B2', self.Nodes)
        self.assertIn('C', self.Nodes)

        #Validade that the progression betwhen nodes
        self.assertIs(   self.Nodes['A'], self.Nodes['C'].next[0])
        self.assertNotIn(self.Nodes['A'], self.Nodes['A'].next)
        self.assertNotIn(self.Nodes['A'], self.Nodes['B1'].next)
        self.assertNotIn(self.Nodes['A'], self.Nodes['B2'].next)

        self.assertIs(   self.Nodes['B1'], self.Nodes['A'].next[0])
        self.assertNotIn(self.Nodes['B1'], self.Nodes['B1'].next)
        self.assertNotIn(self.Nodes['B1'], self.Nodes['B2'].next)
        self.assertNotIn(self.Nodes['B1'], self.Nodes['C'].next)

        self.assertIs(   self.Nodes['B2'], self.Nodes['A'].next[1])
        self.assertNotIn(self.Nodes['B2'], self.Nodes['B2'].next)
        self.assertNotIn(self.Nodes['B2'], self.Nodes['B1'].next)
        self.assertNotIn(self.Nodes['B2'], self.Nodes['C'].next)

        self.assertIs(  self.Nodes['C'], self.Nodes['B1'].next[0])
        self.assertIs(  self.Nodes['C'], self.Nodes['B2'].next[0])
        self.assertNotIn(self.Nodes['C'], self.Nodes['C'].next)
        self.assertNotIn(self.Nodes['C'], self.Nodes['A'].next)

        #Validade input and outputs out nodes
        self.assertIs(self.Nodes['A' ].outputs['0'] , self.Nodes['B1'].inputs['0'])
        self.assertIs(self.Nodes['A' ].outputs['0'] , self.Nodes['B2'].inputs['0'])
        self.assertNotEqual(self.Nodes['A' ].outputs['0'] , self.Nodes['A'].inputs['0'])
        self.assertNotEqual(self.Nodes['A' ].outputs['0'] , self.Nodes['C'].inputs['0'])

        self.assertIs(   self.Nodes['B1'].outputs['0'] , self.Nodes['C'].inputs['0'])
        self.assertNotEqual(self.Nodes['B1'].outputs['0'] , self.Nodes['B1'].inputs['0'])
        self.assertNotEqual(self.Nodes['B1'].outputs['0'] , self.Nodes['C'].inputs['1'])
        self.assertNotEqual(self.Nodes['B1'].outputs['0'] , self.Nodes['A'].inputs['0'])

        self.assertIs(   self.Nodes['B2'].outputs['0'] , self.Nodes['C'].inputs['1'])
        self.assertNotEqual(self.Nodes['B2'].outputs['0'] , self.Nodes['B2'].inputs['0'])
        self.assertNotEqual(self.Nodes['B2'].outputs['0'] , self.Nodes['C'].inputs['0'])
        self.assertNotEqual(self.Nodes['B2'].outputs['0'] , self.Nodes['A'].inputs['0'])

        self.assertIs(   self.Nodes['C' ].outputs['0'] , self.Nodes['A'].inputs['0'])
        self.assertNotEqual(self.Nodes['C' ].outputs['0'] , self.Nodes['C'].inputs['0'])
        self.assertNotEqual(self.Nodes['C' ].outputs['0'] , self.Nodes['B1'].inputs['0'])
        self.assertNotEqual(self.Nodes['C' ].outputs['0'] , self.Nodes['B2'].inputs['0'])

    def test_loop_execution(self, start_value=10):
        Nodes.recipe_reader(self.Nodes, self.Functions, self.recipe)
        
        # Set the initial input value for node A
        for i in self.Nodes['A'].inputs.values():
            i.value = start_value
        
        # Collect outputs from the loop
        expected = [
         start_value,
         start_value,
         start_value,
         start_value*2,
         start_value*2,
         start_value*2,
         start_value*2,
         start_value*4, 
         start_value*4, 
         start_value*4, 
         start_value*4, 
         start_value*8, 
         start_value*8
        ]
        outputs  = []
        for _, node in zip(range(13), self.Nodes['A']):
            outputs.append(node.outputs['0'].value)
        
        self.assertEqual(expected, outputs)
        self.assertEqual(self.Nodes['A'].outputs['0' ].value, start_value*8)
        self.assertEqual(self.Nodes['B1'].outputs['0'].value, self.Nodes['A'].outputs['0'].value/2)
        self.assertEqual(self.Nodes['B2'].outputs['0'].value, self.Nodes['A'].outputs['0'].value/2)
        self.assertEqual(self.Nodes['C'].outputs['0' ].value, start_value*8)

    def test_loop_execution_extended(self):
        self.test_loop_execution(-1.5)
        self.test_loop_execution(-0.5)
        self.test_loop_execution(-0.3)
        self.test_loop_execution(0)
        self.test_loop_execution(0.3)
        self.test_loop_execution(0.5)
        self.test_loop_execution(1.5)

class TestLinkClass(unittest.TestCase):
    def test_link_initialization(self):
        link = Nodes.Link()
        self.assertIsNone(link.value)

        link_with_value = Nodes.Link(42)
        self.assertEqual(link_with_value.value, 42)

    def test_link_value_setter(self):
        link = Nodes.Link()
        link.value = "test"
        self.assertEqual(link.value, "test")

    def test_link_str_representation(self):
        link = Nodes.Link("test_value")
        self.assertIn("test_value", str(link))
        self.assertIn("0x", str(link))

class TestNodeClass(unittest.TestCase):

    def test_self_relationship(self):
        A  = Nodes.Node('A')
        A.next = A
        self.assertIs(A, A.next[0])
        self.assertIs(A, A.prev[0])

    def test_mutual_relationship(self):
        A, B  = Nodes.Node('A'), Nodes.Node('B')
        A.next, B.next = B, A

        self.assertIs(A, B.next[0])
        self.assertIs(A, B.prev[0])
        self.assertIs(B, A.next[0])
        self.assertIs(B, A.prev[0])

    def test_mutual_relationship(self):
        A, B  = Nodes.Node('A'), Nodes.Node('B')
        A.next, B.next = B, A

        self.assertIs(A, B.next[0])
        self.assertIs(A, B.prev[0])
        self.assertIs(B, A.next[0])
        self.assertIs(B, A.prev[0])

    def test_links_betwhen_nodes(self):
        A, B  = Nodes.Node('A'), Nodes.Node('B')

        out = A.reg_output('A_OUT')
        self.assertIsInstance(out, Nodes.Link)

        inp = B.attach_input('B_IN', out)
        self.assertIsInstance(out, Nodes.Link)

        self.assertIs(out, inp)
        self.assertIs(out.value, None)
        
        value = id(out)

        A.output = {'A_OUT': value }

        self.assertIs(A.outputs['A_OUT'].value, value)
        self.assertIs(B.inputs['B_IN'].value, value)
    
    def test_iterable_node(self):
        A, B  = Nodes.Node('A'), Nodes.Node('B')
        A.next = B

        self.assertEqual([n for n in A], [A, B])

