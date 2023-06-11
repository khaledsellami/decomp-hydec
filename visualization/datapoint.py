from typing import Set


class DataPoint:
    """
    A DataPoint is a container that represents either an atom (class or microservice) or a microservice.
    A DataPoint is basically a Tree implementation where each DataPoint can have children and a parent.
    In this case leafs (data points without children) are the atoms"""

    def __init__(self, dp_id=None):
        self.id = dp_id
        self.children = set()
        self.radius = 1  # need for the representation later. Its value corresponds
        self.parent = None

    def __repr__(self):
        r = ""
        if self.id is not None:
            r += "id = " + str(self.id) + " ,\n"
        if len(self.children) > 0:
            r + "children = \n"
            for c in self.children:
                r += "  [" + str(c) + "]\n"
        return r

    def __str__(self):
        r = ""
        if self.id is not None:
            r += "id = " + str(self.id)
        return r

    def add_children(self, children: Set):
        self.children = self.children.union(children)
        self.radius = max(len(self.children), 1)

    def to_circlify(self):
        """
        Returns the data of the current node and its children as a suitable representation for the circlify package
        """
        if self.id is None and len(self.children) == 0:
            return self.radius
        else:
            data = {"datum": self.radius}
            if self.id is not None:
                data["id"] = self.id
            if len(self.children) > 0:
                data["children"] = [c.to_circlify() for c in self.children]
        return data