import numpy as np
from collections import deque
from collections import namedtuple
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

# ra = Rectangle(3., 3., 5., 5.)
# rb = Rectangle(1., 1., 4., 3.5)
# intersection here is (3, 3, 4, 3.5), or an area of 1*.5=.5

class Graph:
    def __init__(self):
        self.graph_dict={}
    def addEdge(self,node,neighbour):  
        if node not in self.graph_dict:
            self.graph_dict[node]=[neighbour]
        else:
            self.graph_dict[node].append(neighbour)
    
    def neighbours(self , node) :
        if node not in self.graph_dict:
            return []
        else:
            return self.graph_dict[node]
    def __del__(self):
        self.graph_dict={}
        # print ("Graph destroyed");

class MaxFlow:
    def __init__(self, n):
        self.n = n
        self.graph = Graph()
        self.cap = np.zeros([n , n] , dtype = float)
        self.INF = 10000000
    
    def addEdge(self , node , neighbour , Q):
        self.graph.addEdge(node , neighbour)
        self.graph.addEdge(neighbour , node)
        self.cap[node][neighbour] += Q
    
    def flow(self, source , sink) :
        q = deque()
        parent = []
        vis = []
        for i in range(self.n):
            parent.append(-1)
            vis.append(0)
        q.append([source , self.INF])
        parent[source] = -1
        vis[source] = 1

        while len(q) > 0 :
            current = q.popleft()
            # print(current)
            if current[0] == sink:
                node = sink
                while node != source:
                    self.cap[parent[node]][node] -= current[1]
                    self.cap[node][parent[node]] += current[1]
                    node = parent[node]
                return current[1]
            else :
                adj = self.graph.neighbours(current[0])
                for child in adj:
                    # print(child)
                    if vis[child] == 1:
                        continue
                    if self.cap[current[0]][child] <= 0:
                        continue
                    f = min(self.cap[current[0]][child] , current[1])
                    parent[child] = current[0]
                    vis[child] = 1
                    q.append([child , f])
        return 0
    
    def getMaxFlow(self , source , sink) :
        res = 0.0
        while 1 :
            f = self.flow(source , sink)
            if f == 0:
                break
            # print(f)
            res += f
        return res
        
    def __del__(self):
        del self.graph
        # print ("Object destroyed");

def area(a, b):  # returns None if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    return dx*dy

def inter(P1 , P2):
    ra = Rectangle(P1[2] , P1[1] , P1[4] , P1[3])
    rb = Rectangle(P2[2] , P2[1] , P2[4] , P2[3])
    get = area(ra , rb)
    ar1 = area(ra , ra)
    ar2 = area(rb , rb)
    if ar1 + ar2 - get == 0:
        return 0
    get = get / (ar1 + ar2 - get)
    return get

def FlowMetrics(Generated , GroundTruth) :
    if len(GroundTruth) == 0 :
        return 1
    source = len(Generated) + len(GroundTruth)
    sink = source + 1
    MF = MaxFlow(sink + 1)


    for i in range(len(Generated)):
        MF.addEdge(source , i , 1.0)
        for j in range(len(GroundTruth)):
            if Generated[i][0] != GroundTruth[j][0]:
                continue
            intersection = inter(Generated[i] , GroundTruth[j])
            if intersection == 0:
                continue
            # print(i,j,intersection)
            MF.addEdge(i, j + len(Generated) , intersection)
    
    for i in range(len(GroundTruth)) :
        MF.addEdge(i + len(Generated) , sink , 1.0)
    
    metric = MF.getMaxFlow(source , sink) / len(GroundTruth)
    del MF
    return metric

def bipartiteMetrics(Generated , GroundTruth) :
    if len(GroundTruth) == 0 :
        return 1
    eps = 0.4
    source = len(Generated) + len(GroundTruth)
    sink = source + 1
    MF = MaxFlow(sink + 1)
    # print(sink + 1)

    for i in range(len(Generated)):
        MF.addEdge(source , i , 1.0)
        for j in range(len(GroundTruth)):
            if Generated[i][0] != GroundTruth[j][0]:
                continue
            intersection = inter(Generated[i] , GroundTruth[j])
            if intersection < eps:
                continue
            MF.addEdge(i, j + len(Generated) , 1.0)
    
    for i in range(len(GroundTruth)) :
        MF.addEdge(i + len(Generated) , sink , 1.0)
    
    metric = MF.getMaxFlow(source , sink) / len(GroundTruth)
    del MF
    return metric

# @params -> Generated -> OCR list for the generated output image
#         -> GroundTruth -> OCR list for the Ground Truth image
# returns list of two float values
# multiply by 100 to get percentage
# format data -> [char character , r1 , r2 , c1 , c2]

def Metrics(Generated , GroundTruth) :
    bipartiteMetric = bipartiteMetrics(Generated , GroundTruth)
    FlowMetric = FlowMetrics(Generated , GroundTruth)
    # print(bipartiteMetric)
    return [bipartiteMetric , FlowMetric] 

# Metrics()


            

