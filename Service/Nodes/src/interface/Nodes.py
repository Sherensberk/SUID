from collections import deque

def reg_node(node_registry, _id):
    _id = str(_id)

    if _id not in node_registry:
        n = Node(_id)
        node_registry[_id] = n 
    
    return node_registry[_id]

def recipe_reader(node_registry, callback_registry, recipe):
    for n in recipe:
        
        node = reg_node(node_registry, n['id'])
        node.callback = callback_registry.get(n['callback'])

        nexts = []
        for c in n['connections']:
            (out_id, inp_id) = tuple(c.items())[0]
            
            next_node = reg_node(node_registry, inp_id.split(':')[0])
            nexts.append(next_node)

            link = node.reg_output(out_id.split(':')[-1])
            next_node.attach_input(inp_id.split(':')[-1], link)

        node.next = nexts

class Link:
    def __init__(self, value=None) -> None:
        self.__value = value        
    @property
    def value(self):
        return self.__value
    
    @value.setter
    def value(self, value):
        self.__value = value

    def __str__(self) -> str:
        return f"[0x{str(id(self))[-4:]}] {self.value} "
    
    def __repr__(self) -> str:
        return self.__str__()

class Node:
    def __init__(self, name) -> None:
        self.name=name
        self.outputs = {}
        self.inputs = {}
        self.__next_nodes = []
        self.__prev_nodes = []
        self.callback_executions=0
        self.__callback = None

    @property
    def next(self):
        return self.__next_nodes
    
    @next.setter
    def next(self, nodes):
        if not isinstance(nodes, (list, tuple)):
            nodes = [nodes]
        self.__next_nodes = nodes
        for node in nodes:
            node.prev = self

    @property
    def prev(self):
        return self.__prev_nodes
    
    @prev.setter
    def prev(self, value):
        if not isinstance(value, list):
            value = [value]
        self.__prev_nodes.extend(value)

    @property
    def callback(self):
        return self.__callback

    @callback.setter
    def callback(self, value):
        self.__callback = value

    def attach_input(self, _id, value:Link):
        if _id not in self.inputs:
            self.inputs[_id] = value
        return self.inputs[_id]

    def reg_output(self, _id):
        if _id not in self.outputs:
            self.outputs[_id] = Link()
        return self.outputs[_id]

    @property
    def output(self):
        return self.outputs
    
    @output.setter
    def output(self, value):
        for (out_id, out_val) in value.items():
            self.outputs.get(out_id).value = out_val
        
    def process(self):
        if self.__callback is not None:
            self.callback_executions += 1
            self.output = self.callback(self)

    def __iter__(self):
        stack = deque([self])

        while stack:
            node = stack.popleft()
            node.process()
            yield node
            for next_node in node.next:
                if next_node not in stack:
                    if len(next_node.inputs) == 1:
                        stack.appendleft(next_node)
                    else:
                        stack.append(next_node)
    
    def __str__(self) -> str:
        # return f"{self.inputs}|>> [|{self.callback_executions}|{self.name}] >>|{self.outputs}"
        return f"[|{self.callback_executions}|{self.name}]"
    
    def __repr__(self) -> str:
        return self.__str__()


class grpcNode(Node):
    def __init__(self,  id, stub, callback, connections) -> None:
        super().__init__(id)
        self._id = id
        self.stub = stub
        self.callback = callback
        self.connections = connections

    def update_connections(self, node_register):
        nexts = []
        for c in self.connections:
            next_node = node_register[c.input.parent]
            nexts.append(next_node)
            link = self.reg_output(c.output.id)
            next_node.attach_input(c.input.id, link)
        self.next = nexts

if __name__ == '__main__':
    raise NotImplemented
