import numpy as np

class Point:
    def __init__(self, x, y):
        """
        a point in euclidean distance graph
        """
        self.x = x
        self.y = y
    
    def euclidean_distance(self, point):
        """
        calculate euclidean distance between 2 points
        """
        return ((self.y - point.y)**2 + (self.x - point.x)**2) ** 1/2
    
    def get(self, replace_x=None):
        """
        get euclidean coordinates (x, y) as a tuple
        
        if replace_x has a value, replace the x coordinate with that
        """
        
        if replace_x:
            return (replace_x, self.y)
        else:
            return (self.x, self.y)
    
    
    def get_point(self, replace_x=None):
        """
        get euclidean coordinates (x, y) as a tuple
        
        if replace_x has a value, replace the x coordinate with that
        """
        if replace_x:
            return Point(replace_x, self.y)
        else:
            return Point(self.x, self.y)
        
        
    def __str__(self):
        return f"x: {self.x}\ny: {self.y}"
    
    @staticmethod
    def np_array_to_Point(np_array):
        return Point(np_array[0], np_array[1])
    
    
class Line:
    def __init__(self, point_1, point_2):
        """
        a line in euclidean distance graph
        The equation for the line in the form of y = mx + c is calculated
        takes as input 2 point objects and finds the straight line that connects them
        """
        self.m = (point_1.y - point_2.y) / (point_1.x - point_2.x) # gradient
        self.c = point_1.y - (self.m * point_1.x) # y intersect
    
    def intersection(self, line):
        """
        find where this line intersects with another line
        calculation from: https://stackabuse.com/solving-systems-of-linear-equations-with-pythons-numpy/
        returns: Point object
        """
        A = np.array([[1, -self.m],
                      [1, -line.m]])
        
        b = np.array([self.c, line.c])
        solution = np.linalg.solve(A, b)

        y = solution[0]
        x = solution[1]
        
        return Point(x, y)
    
    
    def solve(self, x):
        """
        find the y value when x is a certain value
        """
        return int( self.m*x + self.c )
    
    def __str__(self):
        return f"y = {self.m}x + {self.c}"