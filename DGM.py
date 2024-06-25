# CLASS DEFINITIONS FOR NEURAL NETWORKS USED IN DEEP GALERKIN METHOD

#%% import needed packages
import tensorflow as tf #import tensorflow library

#%% LSTM-like layer used in DGM (see Figure 5.3 and set of equations on p. 45) - modification of Keras layer class

class LSTMLayer(tf.keras.layers.Layer):
    
    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, output_dim, input_dim, trans1 = "tanh", trans2 = "tanh"):
        '''
        Args:
            input_dim (int):       dimensionality of input data
            output_dim (int):      number of outputs for LSTM layers
            trans1, trans2 (str):  activation functions used inside the layer; 
                                   one of: "tanh" (default), "relu" or "sigmoid"
        
        Returns: customized Keras layer object used as intermediate layers in DGM
        '''        
        
        # create an instance of a Layer object (call initialize function of superclass of LSTMLayer)
        super(LSTMLayer, self).__init__()
        
        # add properties for layer including activation functions used inside the layer  
        self.output_dim = output_dim  # Set the output dimension
        self.input_dim = input_dim  # Set the input dimension
        
        if trans1 == "tanh":  # Set the first transformation function
            self.trans1 = tf.nn.tanh
        elif trans1 == "relu":
            self.trans1 = tf.nn.relu
        elif trans1 == "sigmoid":
            self.trans1 = tf.nn.sigmoid
        
        if trans2 == "tanh":  # Set the second transformation function
            self.trans2 = tf.nn.tanh
        elif trans2 == "relu":
            self.trans2 = tf.nn.relu
        elif trans2 == "sigmoid":
            self.trans2 = tf.nn.relu
        
        ### define LSTM layer parameters (use Xavier initialization)
        # u vectors (weighting vectors for inputs original inputs x)
        self.Uz = self.add_weight(name="Uz", shape=[self.input_dim, self.output_dim],
                                  initializer=tf.keras.initializers.GlorotNormal())  # Weight matrix for Z gate
        self.Ug = self.add_weight(name="Ug", shape=[self.input_dim ,self.output_dim],
                                  initializer=tf.keras.initializers.GlorotNormal())  # Weight matrix for G gate
        self.Ur = self.add_weight(name="Ur", shape=[self.input_dim, self.output_dim],
                                  initializer=tf.keras.initializers.GlorotNormal())  # Weight matrix for R gate
        self.Uh = self.add_weight(name="Uh", shape=[self.input_dim, self.output_dim],
                                  initializer=tf.keras.initializers.GlorotNormal())  # Weight matrix for H gate
        
        # w vectors (weighting vectors for output of previous layer)        
        self.Wz = self.add_weight(name="Wz", shape=[self.output_dim, self.output_dim],
                                  initializer=tf.keras.initializers.GlorotNormal())  # Weight matrix for Z gate (previous output)
        self.Wg = self.add_weight(name="Wg", shape=[self.output_dim, self.output_dim],
                                  initializer=tf.keras.initializers.GlorotNormal())  # Weight matrix for G gate (previous output)
        self.Wr = self.add_weight(name="Wr", shape=[self.output_dim, self.output_dim],
                                  initializer=tf.keras.initializers.GlorotNormal())  # Weight matrix for R gate (previous output)
        self.Wh = self.add_weight(name="Wh", shape=[self.output_dim, self.output_dim],
                                  initializer=tf.keras.initializers.GlorotNormal())  # Weight matrix for H gate (previous output)
        
        # bias vectors
        self.bz = self.add_weight(name="bz", shape=[1, self.output_dim])  # Bias for Z gate
        self.bg = self.add_weight(name="bg", shape=[1, self.output_dim])  # Bias for G gate
        self.br = self.add_weight(name="br", shape=[1, self.output_dim])  # Bias for R gate
        self.bh = self.add_weight(name="bh", shape=[1, self.output_dim])  # Bias for H gate
    
    # main function to be called 
    def call(self, S, X):
        '''Compute output of a LSTMLayer for a given inputs S,X .    

        Args:            
            S: output of previous layer
            X: data input
        
        Returns: customized Keras layer object used as intermediate layers in DGM
        '''   
        
        # compute components of LSTM layer output (note H uses a separate activation function)
        Z = self.trans1(tf.add(tf.add(tf.matmul(X, self.Uz), tf.matmul(S, self.Wz)), self.bz))  # Compute Z gate output
        G = self.trans1(tf.add(tf.add(tf.matmul(X, self.Ug), tf.matmul(S, self.Wg)), self.bg))  # Compute G gate output
        R = self.trans1(tf.add(tf.add(tf.matmul(X, self.Ur), tf.matmul(S, self.Wr)), self.br))  # Compute R gate output
        
        H = self.trans2(tf.add(tf.add(tf.matmul(X, self.Uh), tf.matmul(tf.multiply(S, R), self.Wh)), self.bh))  # Compute H gate output
        
        # compute LSTM layer output
        S_new = tf.add(tf.multiply(tf.subtract(tf.ones_like(G), G), H), tf.multiply(Z, S))  # Compute new state S_new
        
        return S_new

#%% Fully connected (dense) layer - modification of Keras layer class
   
class DenseLayer(tf.keras.layers.Layer): # f function line in page 45
    
    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, output_dim, input_dim, transformation=None):
        '''
        Args:
            input_dim:       dimensionality of input data
            output_dim:      number of outputs for dense layer
            transformation:  activation function used inside the layer; using
                             None is equivalent to the identity map 
        
        Returns: customized Keras (fully connected) layer object 
        '''        
        
        # create an instance of a Layer object (call initialize function of superclass of DenseLayer)
        super(DenseLayer, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        
        ### define dense layer parameters (use Xavier initialization)
        # w vectors (weighting vectors for output of previous layer)
        self.W = self.add_weight(name="W", shape=[self.input_dim, self.output_dim],
                                 initializer=tf.keras.initializers.GlorotNormal())
        
        # bias vectors
        self.b = self.add_weight(name="b", shape=[1, self.output_dim])
        
        if transformation:
            if transformation == "tanh":
                self.transformation = tf.tanh
            elif transformation == "relu":
                self.transformation = tf.nn.relu
        else:
            self.transformation = transformation
    
    
    # main function to be called 
    def call(self, X):
        '''Compute output of a dense layer for a given input X 

        Args:                        
            X: input to layer            
        '''
        
        # compute dense layer output
        S = tf.add(tf.matmul(X, self.W), self.b)
                
        if self.transformation:
            S = self.transformation(S)
        
        return S

#%% Neural network architecture used in DGM - modification of Keras Model class
    
class DGMNet(tf.keras.Model):
    
    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, layer_width, n_layers, input_dim, final_trans=None):
        '''
        Args:
            layer_width: 
            n_layers:    number of intermediate LSTM layers
            input_dim:   spaital dimension of input data (EXCLUDES time dimension)
            final_trans: transformation used in final layer
        
        Returns: customized Keras model object representing DGM neural network
        '''  
        
        # create an instance of a Model object (call initialize function of superclass of DGMNet)
        super(DGMNet, self).__init__()
        
        # define initial layer as fully connected 
        # NOTE: to account for time inputs we use input_dim+1 as the input dimensionality
        self.initial_layer = DenseLayer(layer_width, input_dim + 1, transformation="tanh")
        
        # define intermediate LSTM layers
        self.n_layers = n_layers
        self.LSTMLayerList = []
                
        for _ in range(self.n_layers):
            self.LSTMLayerList.append(LSTMLayer(layer_width, input_dim + 1))
        
        # define final layer as fully connected with a single output (function value)
        self.final_layer = DenseLayer(1, layer_width, transformation=final_trans)
    
    
    # main function to be called  
    def call(self, inputs):
        '''            
        Args:
            inputs: tuple of inputs (t, y, delta_a, delta_b)

        Run the DGM model and obtain fitted function value at the inputs                
        '''  
        t, y, delta_a, delta_b = inputs
        
        # define input vector as concatenated time-space pairs
        X = tf.concat([t, y, delta_a, delta_b], 1)
        
        # call initial layer
        S = self.initial_layer.call(X)
        
        # call intermediate LSTM layers
        for i in range(self.n_layers):
            S = self.LSTMLayerList[i].call(S, X)
        
        # call final layer
        result = self.final_layer.call(S)
        
        return result
