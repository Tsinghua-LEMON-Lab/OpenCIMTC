
class OneOnOneAlgorithm(object):
    
    def __init__(self,node_weight,XB_size):
        self.node_weight = node_weight
        self.XB_size = XB_size
        
    def run(self):
        '''
        place nodes through one-to-one manner
        '''
        all_node_addr = []
        
        for node_name in self.node_weight.keys():
            node_list_per_XB = []
            node_addr = {node_name:[0,0,self.node_weight[node_name][1],self.node_weight[node_name][0]]}
            node_list_per_XB.append(node_addr)
            all_node_addr.append(node_list_per_XB)
        return all_node_addr