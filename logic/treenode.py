class TreeNode(object):
    def __init__(self, element: int):
        self.element = element
        self.values = dict()
        self.children = list()
        self.parent = None

    def add_child(self, element=None):
        if isinstance(element, TreeNode):
            self.children.append(element)
            element.parent = self
            return element
        else:
            node = TreeNode(element)
            self.children.append(node)
            node.parent = self
            return node

    def is_leaf(self):
        return len(self.children)==0