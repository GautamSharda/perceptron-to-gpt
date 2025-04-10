import math
from graphviz import Digraph

class Value:
    def __init__(self, value, children=None):
        self.value = value
        self.op = None
        self.children = children
        self.gradient = 0.0
        self.right = False
    
    def __add__(self, other_value, op='+'):
        new_value = self.value + other_value.value
        self.op = op
        other_value.op = op
        other_value.right = True
        return Value(new_value, [self, other_value])

    def __mul__(self, other_value, op='*'):
        new_value = self.value * other_value.value
        self.op = op
        other_value.op = op
        other_value.right = True
        return Value(new_value, [self, other_value])

    def __sub__(self, other_value, op='-'): # Makes sense to to do this via add too, same with MUL in DIV
        new_value = self.value - other_value.value
        self.op = op
        other_value.op = op
        other_value.right = True
        return Value(new_value, [self, other_value])
    
    def __truediv__(self, other_value, op='/'):
        new_value = self.value / other_value.value
        self.op = op
        other_value.op = op
        other_value.right = True
        return Value(new_value, [self, other_value])

    def __abs__(self):
        self.op = '|'
        return Value(self.value*-1 if self.value < 0 else self.value, [self, None])
    
    def backward(self, other_value=None, chain_rule_grad=1.0):
        if self.op == "+":
            self.gradient += 1.0*chain_rule_grad
        elif self.op == "-":
            if self.right:
                self.gradient += -1.0*chain_rule_grad
            else:
                self.gradient += 1.0*chain_rule_grad
        elif self.op == '*':
            self.gradient += other_value.value*chain_rule_grad
        elif self.op == '/':
            if self.right:
                self.gradient += ((-other_value.value)/(self.value**2))*chain_rule_grad
            else:
                self.gradient += (1 / other_value.value)*chain_rule_grad
        elif self.op == "|":
            self.gradient += -1.0*chain_rule_grad if self.value < 0 else 1.0*chain_rule_grad
        else: # Op will be None for root
            self.gradient += 1.0
        if not self.children:
            return
        child_1, child_2 = self.children
        child_1.backward(child_2, self.gradient)
        child_2.backward(child_1, self.gradient) if child_2 else None

def make_computation_graph(root_node, filename="out/computation_graph"):
    nodes_obj, edges_obj = set(), set()
    visited_ids = set()

    def build_graph_obj(v):
        if id(v) in visited_ids:
            return
        visited_ids.add(id(v))
        nodes_obj.add(v)
        if hasattr(v, 'children') and v.children:
            for child in v.children:
                if child: # Skip None children (like in abs)
                    edges_obj.add((child, v))
                    build_graph_obj(child)

    build_graph_obj(root_node)

    dot = Digraph(format='png', graph_attr={'rankdir': 'LR'})
    op_nodes_created = set() # Track operation nodes to avoid duplicates

    for n in nodes_obj:
        uid = str(id(n))
        val_str = f"{n.value:.4f}" if isinstance(n.value, (int, float)) and not math.isnan(n.value) else str(n.value)
        grad_str = f"{n.gradient:.4f}" if isinstance(n.gradient, (int, float)) and not math.isnan(n.gradient) else str(n.gradient)
        label_str = f"val {val_str} | grad {grad_str}"
        dot.node(name=uid, label=label_str, shape='record')

    for child, parent in edges_obj:
        child_uid = str(id(child))
        parent_uid = str(id(parent))

        op_label = getattr(child, 'op', None) # Get op from child

        if op_label:
            op_uid = parent_uid + '_op_' + op_label

            # Create the operation node only once per parent-operation pair
            if op_uid not in op_nodes_created:
                dot.node(name=op_uid, label=op_label, shape='ellipse')
                dot.edge(op_uid, parent_uid)
                op_nodes_created.add(op_uid)

            dot.edge(child_uid, op_uid)
        else:
            # If child has no 'op', maybe it's an initial value?
            # Draw a direct edge for structure, though less informative.
            # This case might not happen if all ops set child.op correctly.
            dot.edge(child_uid, parent_uid)
            # print(f"Warning: Child node {child_uid} has no 'op' for edge to parent {parent_uid}")


    try:
        dot.render(filename, view=True, cleanup=True)
        print(f"Graph rendered to {filename}.png and opened.")
        input("Press Enter to continue training...")
    except Exception as e:
        print(f"Error rendering graph (is graphviz installed and in PATH?): {e}")
        print("\nGraphviz Source:\n----------------")
        print(dot.source)
        print("----------------\n")
        input("Error displaying graph. Press Enter to continue training...")
