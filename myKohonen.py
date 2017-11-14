# BSD license 
#
# Date : 2017 
# Author : Boltenhagen Mathilde inspired by Mathieu Lefort
# python 2.7 for cPickle
#
#Sources : 
# Mathieu Lefort
# https://samos.univ-paris1.fr/IMG/pdf_Porvoo_Kohonen_Data_Analysis_V3-2.pdf
# https://samos.univ-paris1.fr/archives/ftp/preprints/samos132.pdf
# http://blog.yhat.com/posts/self-organizing-maps-2.html
#



import gzip
import numpy
import matplotlib.pyplot as plt
import cPickle


'''
computes distance from entry to neuron
@param x (numpy array) : entry 
'''
def distance(weights, x):
    sum = numpy.sum((numpy.subtract(x, weights)) ** 2)
    return numpy.math.sqrt(sum)

'''
Changes weights with Kohonen rule
@param eta (float): 
@param type (string): hexa or rect
@param sigma (float): size of neigbours 
@param posxW (int): pos x of the neuron winner, which is the closest to the entry
@param posYW (int):  pos y of the neuron winner, which is the closest to the entry
@param x (numpy array) : entry

'''
def learn(weigths, posx, posy, eta, sigma, posxW, posyW, x, type):
    #hexagonal distance
    if (type == 'hexa'):
        hexagonal_distance = (abs(posyW - posy) + abs(posyW + posxW - posy - posx) + abs(
            posxW - posx)) / 2
        equa_gausienne_hexagonale = numpy.math.exp(-(hexagonal_distance / ((2 * (sigma ** 2)))))

        #change weights
        weigths[:] += ((x[:] - weigths[:]) * eta * equa_gausienne_hexagonale)
    else:
        # rectangle distance
        equa_gaussienne = numpy.math.exp(
            -(((posxW - posx) ** 2 + (posyW - posy) ** 2) / ((2 * (sigma ** 2)))))

        # change weights
        weigths[:] += ((x[:] - weigths[:]) * eta * equa_gaussienne)
    return weigths


'''
Initialises the maps with zero for the Output and with random values for the weights
@param inputsize (int, int) : size of the vectors of inputs
@param mapsize (int, int) : size of the map of neurons
@return mapOutput : zero numpy map
@return mapWeights : random numpy map 
'''
def init_network (inputsize,mapsize):
    #create an zero map
    mapOutput=numpy.zeros((mapsize[0],mapsize[1]))
    mapWeights=[]
    for posx in range(0,mapsize[0]):
        mline = []
        for posy in range(0,mapsize[1]):
	    #create random weights for each neurons
            weights = numpy.random.random(inputsize)
            mline.append(weights)
        mapWeights.append(mline)
    return (mapOutput,mapWeights)

'''
Computes the Mean Squared Error 
@param data : all the entries
@param mapOutput : the map of the Output
@param mapWeights : the last computed map of the weights
@mapsize (int, int) : size of the map of the neurons
'''
def error (data, mapOutput, mapWeights, mapsize):
    s=0
    for x in data:
        for posx in range(0,mapsize[0]):
            for posy in range(0, mapsize[1]):
		#compute the distance from the weights of the neurons to the entry
                mapOutput[posx][posy]=distance(mapWeights[posx][posy],x)
        s += numpy.min(mapOutput) ** 2
    return s/data.shape[0]

'''
Self-Organized Map : KOHONEN
@param inputesize (int,int) : size of the vectors of inputs
@param mapsize (int, int) : size of the map of neurons
@param eta : apprenticeship rate
@param type (string): hexa or rect
@param sigma (float): size of neigbours 
@param sizelearning (int) : number of iteration
@param data : map 
'''
def SOM (inputsize, mapsize,sigma, eta, type, sizelearning,data ) :

    #initialize the output map with zeros and weights map with random values
    mapOutput,mapWeights=init_network(inputsize,mapsize)

    #start the loop 
    for i in range(1,sizelearning):
        #get a random entry
        index = numpy.random.randint(len(data))
        x = data[index]

        #compute the distance to the neurons from the entry 
        for posx in range(0,mapsize[0]):
            for posy in range(0, mapsize[1]):
                mapOutput[posx][posy]=distance(mapWeights[posx][posy],x)
	
	#get the winner neuron
        posxW,posyW = numpy.unravel_index(numpy.argmin(mapOutput),mapsize)

	#adjuste weight depending on the winner neuron
        for posx in range(0, mapsize[0]):
            for posy in range(0, mapsize[1]):
                mapWeights[posx][posy]= learn(mapWeights[posx][posy], posx, posy, eta, sigma, posxW, posyW, x, type)

    #return map of neurons with the weights
    return  (mapOutput,mapWeights)

'''
shows the weights of each neurons
@mapsize (int, int) : size of the map of neurons
@inputsize (int,int) : size of the vectors of inputs
@mapweight : 
'''
def plot (mapsize,inputsize,mapWeights):
    for i in range(0,mapsize[0]):
        for j in range(0,mapsize[1]):
            plt.subplot(mapsize[0],mapsize[1],(i*mapsize[0]+1)+j)
            plt.imshow(mapWeights[i][ j], interpolation='nearest', vmin=0, vmax=1, cmap='binary')
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                                wspace=0.35)
    plt.show()



###############################################################################
##                EXAMPLE OF USING WITH MNIST DATA                        #####
###############################################################################

data =cPickle.load(gzip.open('../mnist.pkl.gz'))

w=SOM ((28,28),(9,9),2, 0.05, 'rect', 50000,data )
plot ((9,9),(28,28),w[1])
print("Mean Squared Error: " + error(data, w[0], w[1], (9,9)))
