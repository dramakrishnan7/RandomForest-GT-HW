from util import entropy, information_gain, partition_classes
import numpy as np 
import statistics
import ast

class node(object):
    def __init__(self):
       
        self.column =None
        self.value =None
        self.children = {}
        self.parent = None
        self.predicted_value = None


class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        #self.tree = []
        self.root = node()
        self.last_node2 = node()
        self.prev_node2= node()
        self.tree2 = {}
        self.lane = ''
        
        #below vars only used for checks
        self.leaf_num = 0
        self.saved_nodes = []
        self.previous_node_info = ''
        
        #pass
    
    def learn(self, X, y):
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in utils.py to train the tree
        
        
        
        
        
        
        
        gain = 0
        max_gain = 0
        max_gain_col = None
        max_gain_split_val = None
     
        X_2np = np.asarray(X)
        
        #get the # of cols
        
        num_cols = 1
        try:
            num_cols = X_2np.shape[1]
        except IndexError:
            num_cols = 1
            
        
        for attr_col in range(0,num_cols):
            
            if (attr_col in self.saved_nodes):
               continue
            
            #Check if the col is alpha or float/int
            try:
                float_n = float(X[0][attr_col])
            except ValueError:
                float_n = X[0][attr_col]
                            
            #print(attr_col,float_n,isinstance(float_n,int),isinstance(float_n,float),isinstance(float_n,str) )
            #Use the median if the column value is float or integer
            #Use the max occurance of the column value if it is string
            
            if isinstance(float_n,int) or isinstance(float_n,float):
               X_2npf = X_2np[:,attr_col].astype(np.float)
               median_value = float(np.mean(X_2npf,axis=0))
                
            if isinstance(float_n,str):
                set_aa = list(X_2np[:,attr_col].flatten())
                median_value =  max(set(set_aa), key=set_aa.count)
               
            
            #print(attr_col,median_value)
            
            X_left1, X_right1, y_left1, y_right1 = partition_classes(X_2np,y, attr_col, median_value)
            gain = information_gain(y,[y_left1,y_right1])
            
            #print('GAIN',gain)
            #print('X',X)
            #print('X_R', X_right1)
            #print('X_L', X_left1)
            #if (gain > .95):
                #break
            
            if (gain > max_gain):
                max_gain = gain
                max_gain_col = attr_col
                max_gain_split_val = median_value
            if (max_gain_col==None):
                max_gain_col = attr_col
                max_gain_split_val = median_value
            
        X_left1, X_right1, y_left1, y_right1 = partition_classes(X,y, max_gain_col, max_gain_split_val)    
        
        #print(max_gain_col, max_gain_split_val) 
        #print(X_left1, X_right1, y_left1, y_right1) 
        
        #very little information gain
        #do not split
        #keep the predicted value of the whole set
        #still keeping the max_gain_col and max_gain_split_val
        #If root node or any node goes below 0.05, use the predicted value 
        #children will be empty
        
        if (max_gain <= 0.00000 ):
             
            #print('In 0.05')
            #print(max_gain_col, max_gain_split_val,self.root.column)
            #print('BEFORE','ROOT',self.root.column,'Last_Node2',self.last_node2.column,'Prev_Node',self.previous_node.column)
            #self.saved_nodes.append(max_gain_col)
            #print(X_left1, X_right1)
            #print(y_left1, y_right1) 
            
            y_val1 = np.bincount(y)
            predicted_val = np.nonzero(y_val1)[0][0]
            
            if (self.root.column==None):
                #print('setting root',max_gain_col,max_gain_split_val)
                #self.root.column= max_gain_col
                #self.root.value = max_gain_split_val
                self.root.predicted_value = predicted_val
                self.last_node2 = self.root
                self.previous_node=self.root
            else:
                #self.last_node2.column=max_gain_col
                #self.last_node2.value=max_gain_split_val
                self.last_node2.predicted_value = predicted_val
            if not(self.last_node2 == self.root):
                self.last_node2 = self.last_node2.parent
                #print('Resetting last_node')
                #print('Resetting last_node',self.last_node2.column)
                              
                
                #self.last_node2 = self.last_node2.children[self.lane]
                #self.previous_node.predicted_value = predicted_val
                #self.last_node2.column = max_gain_col
                #self.last_node2.value = max_gain_split_val
            #print('AFTER','ROOT',self.root.column,'Last_Node2',self.last_node2.column,'Prev_Node',self.previous_node.column)
            
            return
                    
        
        if (max_gain >= 0.99999 ):
            
            #Predicted value before split
            #print('No more splits- great gains')
            #print(max_gain_col, max_gain_split_val,self.root.column)
            #print('ROOT',self.root.column,'Last_Node2',self.last_node2.column,'Prev_Node',self.previous_node.column)
            #self.saved_nodes.append(max_gain_col)
            #print(X_left1, X_right1)
            #print(y_left1, y_right1) 
            
            y_val1 = np.bincount(y)
            #predicted_val_l = np.nonzero(y_val1)[0][0]
            #predicted_val_r = np.nonzero(y_val1)[0][0]
            
            #Predicted value for left split
            if ( len(y_left1)>=len(y_right1)):
                y_left_val1 = np.bincount(y_left1)
                predicted_val_l = np.nonzero(y_left_val1)[0][0]
                predicted_val_r = 1-predicted_val_l
            else:
                y_right_val1 = np.bincount(y_right1)
                predicted_val_r = np.nonzero(y_right_val1)[0][0]
                predicted_val_l = 1-predicted_val_r
            
            #print(predicted_val_l,predicted_val_r)
            
            if (self.root.column==None):
                #if from root node do not set the predicted value
                self.root.column= max_gain_col
                self.root.value = max_gain_split_val
                self.last_node2 = self.root
                self.previous_node=self.root
            else:
                
                self.last_node2.column=max_gain_col
                self.last_node2.value=max_gain_split_val
                
                #print('AFTER','ROOT',self.root.column,'Last_Node2',self.last_node2.column,'Prev_Node',self.previous_node.column)
          
            #set the left and right node 
            #set the predicted value for both nodes
            #do not set the column/value , since there will be bo further splits
                
            root_left_final = node()
            root_left_final.predicted_value = predicted_val_l
            self.last_node2.children['L']= root_left_final
            root_left_final.parent = self.last_node2.children['L']
                
            root_right_final = node()
            root_right_final.predicted_value = predicted_val_r
            self.last_node2.children['R']= root_right_final
            root_right_final.parent = self.last_node2.children['R']
            
            if not(self.last_node2 == self.root):
                #print('Resetting last_node')
                self.last_node2 = self.last_node2.parent 
                #print('Resetting last_node')
                #print('Resetting last_node',self.last_node2.column)
            
            
            return
        
            
        if (0.00000 < max_gain < 0.99999):
            #print('Keep splitting')
            #print("GAIN:",max_gain, "COL",max_gain_col,"VAL", max_gain_split_val,"ROOT",self.root.column)
            #print(self.root.column,self.last_node2.column,self.previous_node.column)
            #self.saved_nodes.append(max_gain_col)
            #print(X_left1, X_right1)
            #print(y_left1, y_right1) 
            
                      
            if (self.root.column==None):
                self.last_node2 = self.root
                self.previous_node=self.root
                #print('AT-ROOT',self.root.column,self.last_node2.column,self.previous_node.column)
            
            self.last_node2.column= max_gain_col
            self.last_node2.value = max_gain_split_val
            self.lane ='L'
            new_node_L = node()
            #print(self.lane,new_node)
            #print('BEFORE:L:','ROOT',self.root.column,'Last_Node2',self.last_node2.column,'Prev_Node',self.previous_node.column)
            self.last_node2.children[self.lane]= new_node_L
            self.previous_node=self.last_node2
            self.last_node2 = self.last_node2.children[self.lane]
            self.last_node2.parent = self.previous_node
            #print('AFTER:L:','ROOT',self.root.column,'Last_Node2',self.last_node2.column,'Prev_Node',self.previous_node.column)
            self.learn(X_left1,y_left1)
            #if not(self.last_node2 == self.root):
            #    self.last_node2 = self.last_node2.parent
            #    print('Resetting last_node')
            #    print('Resetting last_node',self.last_node2.column)
            
            self.lane ='R'
            new_node = node()
            #print('BEFORE:R:','ROOT',self.root.column,'Last_Node2',self.last_node2.column,'Prev_Node',self.previous_node.column)
            #new_node.column= max_gain_col
            #new_node.value= max_gain_split_val
            self.last_node2.children[self.lane]= new_node
            self.previous_node=self.last_node2
            self.last_node2 = self.last_node2.children[self.lane]
            self.last_node2.parent = self.previous_node
            #print('AFTER:R:','ROOT',self.root.column,'Last_Node2',self.last_node2.column,'Prev_Node',self.previous_node.column)
            self.learn(X_right1,y_right1)
            if not(self.last_node2 == self.root):
                self.last_node2 = self.last_node2.parent
                #print('Resetting last_node')
                #print('Resetting last_node',self.last_node2.column)
            
        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a 
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)
        #pass


    def classify(self, record):
        # TODO: classify the record using self.tree and return the predicted label
        #return self.tree[str(record)]
        #Add Plus 
        
        #Start from the root
      
        tree_iter = True
        current_node = self.root
        is_a_num = True
        loop=0
        #print(current_node.predicted_value,self.root.value, self.root.column,current_node.value,current_node.column )
        
        
        
        while (tree_iter):
            loop= loop+1
            if (loop>20):
                #print("Inloop",record,current_node.predicted_value)
                if (current_node.predicted_value == 1):
                    return 0;
                else:
                    return 1;
            
            #if we have the predicted_value from the node return 
            #print("CLASSIFY", current_node.predicted_value,self.root.column,current_node.column,current_node.value )
            #print(record,'looping')
            if (current_node.column == None):
                #print('IN-NONE',current_node.parent.column,current_node.predicted_value)
                tree_iter = False
                return current_node.predicted_value
            
            
            #print("CLASSIFY", current_node.predicted_value,self.root.column,current_node.column,current_node.value )
            
            try:
                float_v = float(current_node.value)
            except ValueError:
                float_v = current_node.value
                
                
            if isinstance(float_v,int) or isinstance(float_v,float):
                is_a_num = True
            else:
                is_a_num = False
            
            
            
            #print(current_node.predicted_value,current_node.column, current_node.value, current_node.children)
            
            if (is_a_num):
                #print(current_node.predicted_value,'NUMBER',current_node.column,record[int(current_node.column)],current_node.value)
                if (record[int(current_node.column)] <= current_node.value):
                    if (current_node.children.get('L')):
                        #print('IN-L')
                        current_node = current_node.children['L']
                        #print("CLASSIFY", 'PV:',current_node.predicted_value,'ROOT:',self.root.column,'CURR_NODE_COL',current_node.column,'CURR_NODE_VAL',current_node.value )
                        continue
                        #print('DID current node change',current_node)
                    
                else:
                    if (current_node.children.get('R')):
                        #print('IN-R')
                        current_node = current_node.children['R']
                        continue
            else:
                #print('NOT  NUMBER')
                #print(record)
                #print('STATS-CURR-NODE',current_node.predicted_value,current_node.column, current_node.value,)
                if (record[int(current_node.column)] == current_node.value):
                    if (current_node.children.get('L')):
                        current_node = current_node.children['L']
                        continue
                        #print(current_node.predicted_value,current_node.column, current_node.value, current_node.children)
                     
                else:
                    if (current_node.children.get('R')):
                        current_node = current_node.children['R']
                        continue
                    
                        
            #print(current_node.predicted_value,current_node.column, current_node.value, current_node.children)
            
            
        
       
        
    
