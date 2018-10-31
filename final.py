# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 03:22:28 2017

@author: user
"""

from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy import spatial
import random
MUTATION_RATE = 0.1 #0.25
#CHROMOSOME_SIZE = 1000
POPULATION_SIZE = 10
TOURNAMENT_SELECTION_SIZE= (int)(POPULATION_SIZE/4)
NUMB_OF_ELITE_CHROMOSOMES = 2 #(int)(POPULATION_SIZE/4) #populasyonun %10
blank_image = cv2.imread('1.png',0)
height, width = blank_image.shape[:2]
CIRCLE_RADIUS = 40
 
def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
  
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err
def compare_images(imageA, imageB, title):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)
    # setup the figure
#   
#    fig = plt.figure(title)
#    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
#    # show first image
#    ax = fig.add_subplot(1, 2, 1)
#    plt.imshow(imageA)
#    plt.axis("off")
#    # show the second image
#    ax = fig.add_subplot(1, 2, 2)
#    plt.imshow(imageB)
#    plt.axis("off")
#    # show the images
#    print(title)
#    plt.show()
   
    return m  
#similarity = spatial.distance.cosine(img[:,:,0], blank_image[:,:,0])
def circle(image, chromosome):
#    img = np.copy(image)
    cv2.circle(image, chromosome[0], chromosome[1], chromosome[2], -1)
    return image, chromosome
def gene():
    center = (random.randint(0, height), random.randint(0, width))
    radius = random.randint(0, CIRCLE_RADIUS)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return np.array([center, radius, color])
#def chromosome(l):
#    l.append(gene())
#Popülasyon tanımlanması
class Population:
    def __init__(self, size):
        self._chromosomes = []
        i = 0
        while i < size:
            self._chromosomes.append(Chromosome())
            i += 1
          
    def get_chromosomes(self):
        return self._chromosomes
   
    
    def increase(self):
        for each in self._chromosomes[:NUMB_OF_ELITE_CHROMOSOMES]:
            each.inc_genesElite()
        for each in self._chromosomes[NUMB_OF_ELITE_CHROMOSOMES:]:
            each.inc_genes()    
    '''        
    def increase(self):
        for each in range(0,NUMB_OF_ELITE_CHROMOSOMES):
            Elitchro=self.get_chromosomes[each]
            Elitchro.inc_genesElite()
            self.get_chromosomes[each]=Elitchro
        for each in range(NUMB_OF_ELITE_CHROMOSOMES,len(self._chromosomes)):
            self.get_chromosomes[each].inc_genes()
    '''        
#Kromozom tanımlanması
class Chromosome:
    def __init__(self):
#        if not size == None:
#            print('size non degil:',size)
#            self._genes = [None]*size
        self._genes = []
        self._fitness = 0
#        self._genes.append(gene())
#        while len(self._genes) < CHROMOSOME_SIZE:
#            self._genes.append(gene())
    def del_gene(self):
        del self._genes[-1]
        
    def inc_genes(self):
        x=gene()
        self._genes.append(x)
        
    def inc_genesElite(self):
        x=gene()
        fit=self.get_fitness()
        self._genes.append(x)
        while fit < self.get_fitness():
            self.del_gene()
            x=gene()
            self._genes.append(x)
       
    def get_genes(self):
        return self._genes
  
    def get_fitness(self):
        self._fitness = 0
#        blank_image = np.zeros((height,width,3), np.uint8)
        image = np.zeros((height,width), np.uint8)
        for x in range(self._genes.__len__()):
            circle(image, self._genes[x])
#        return mse(blank_image, image)
        return compare_images(blank_image, image,str(self._genes.__len__()))
    def __str__(self):
        return self._genes.__str__()
    
    
#Genetik algoritma
class GeneticAlgorithm:
    @staticmethod
    def evolve(pop):
        return GeneticAlgorithm._mutate_population(GeneticAlgorithm._crossover_population(pop))
   
    @staticmethod
    def _crossover_population(pop):
        crossover_pop = Population(0)
        for i in range(NUMB_OF_ELITE_CHROMOSOMES):
            crossover_pop.get_chromosomes().append(pop.get_chromosomes()[i])
        i = NUMB_OF_ELITE_CHROMOSOMES
        while i < POPULATION_SIZE:
            chromosome1 = GeneticAlgorithm._select_tournament_population(pop).get_chromosomes()[0]
            chromosome2 = GeneticAlgorithm._select_tournament_population(pop).get_chromosomes()[0]
            crossover_pop.get_chromosomes().append(GeneticAlgorithm._crossover_chromosomes(chromosome1, chromosome2))
            i += 1
        return crossover_pop
   
    @staticmethod
    def _mutate_population(pop):
        for i in range(NUMB_OF_ELITE_CHROMOSOMES, POPULATION_SIZE):
            GeneticAlgorithm._mutate_chromosome(pop.get_chromosomes()[i])
        return pop
   
    @staticmethod
    def _crossover_chromosomes(chromosome1, chromosome2):
        crossover_chrom = Chromosome()
#        for x in range(len(chromosome1.get_genes())):
#            crossover_chrom.inc_genes()
        for i in range(len(chromosome1.get_genes())):
            if random.random() < 0.5:
                crossover_chrom.get_genes().append(chromosome1.get_genes()[i].copy())
            else:
                crossover_chrom.get_genes().append(chromosome2.get_genes()[i].copy())
#        print('c1',chromosome1.get_genes()[0])
#        print('c2',chromosome2.get_genes()[0])
#        print('x',crossover_chrom.get_genes()[0])
        return crossover_chrom
   
    @staticmethod
    def _mutate_chromosome(chromosome):
        for i in range(len(chromosome.get_genes())):
            if random.random() < MUTATION_RATE:
                rnd = random.random()
                if rnd >= 0.66:
                    #center
                    center=chromosome.get_genes()[i][0][0],chromosome.get_genes()[i][0][1]
                    if random.random() >= 0.5:
                        center = random.randint(0, height),chromosome.get_genes()[i][0][1]
                    else:
                        center = chromosome.get_genes()[i][0][0],random.randint(0, width)
                    chromosome.get_genes()[i][0]=center   
                elif rnd >= 0.33:
                    #radius
                    chromosome.get_genes()[i][1] = radius = random.randint(0, CIRCLE_RADIUS) #radius
                elif rnd >= 0.00:
                    crnd=random.random()
                    #color
                    color=chromosome.get_genes()[i][2][0],chromosome.get_genes()[i][2][1],chromosome.get_genes()[i][2][2]
                    if crnd>=0.66:
                        color = random.randint(0, 255),chromosome.get_genes()[i][2][1],chromosome.get_genes()[i][2][2]
                    elif crnd>=0.33:
                        color =chromosome.get_genes()[i][2][0], random.randint(0, 255),chromosome.get_genes()[i][2][2]
                    else:
                        color = chromosome.get_genes()[i][2][0],chromosome.get_genes()[i][2][1],random.randint(0, 255)
                    chromosome.get_genes()[i][2]=color    
              
        
    @staticmethod
    def _select_tournament_population(pop):
        tournament_pop = Population(0)
        i = 0
        while i < TOURNAMENT_SELECTION_SIZE:
            tournament_pop.get_chromosomes().append(pop.get_chromosomes()[random.randrange(0, POPULATION_SIZE)])
            i += 1
        tournament_pop.get_chromosomes().sort(key = lambda x: (x.get_fitness()<0,abs(x.get_fitness())))
        return tournament_pop
   
#Ekrana yazdırma fonksiyonu   
def print_population(pop, gen_number):
    print("\n-------------------------------")
    print("Generation #", gen_number, "| Fittest chromosome fitness:", pop.get_chromosomes()[0].get_fitness()) 
    #print("Target Point:", TARGET_POINT_X, TARGET_POINT_Y)
    print("---------------------------------")
    i = 0
    for x in pop.get_chromosomes():
        print("Chromosome #", i, " : ", "|Fitness: ", x.get_fitness())
#        for y in x.get_genes():
#           print("Gene #", i, " : ", y)
        i += 1   
'''   
img = []
pop = Population(POPULATION_SIZE)
for i in range(0, POPULATION_SIZE):
    print('Population Number:',i)
    pop.get_chromosomes()[i].get_genes()
    blank_image = np.zeros((height,width,3), np.uint8)
    for j in range(0, CHROMOSOME_SIZE):
        image, chro = circle(blank_image, pop.get_chromosomes()[i].get_genes()[j])
        img.append(image)
    print('chro: ',i,pop.get_chromosomes()[i].get_fitness())
    cv2.imshow(str(i), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''
pop = Population(POPULATION_SIZE)
pop.increase()  
pop.get_chromosomes().sort(key = lambda x: (x.get_fitness()<0,abs(x.get_fitness())))  
generation_number = 0
print_population(pop, -1)
while pop.get_chromosomes()[0].get_fitness() > 0:
    pop = GeneticAlgorithm.evolve(pop)
    pop.get_chromosomes().sort(key = lambda x: (x.get_fitness()<0,abs(x.get_fitness())))
    print_population(pop, generation_number)
#    print('GENERASYON SAYISI',generation_number)
    generation_number += 1
#    blank_image = np.zeros((height,width,3), np.uint8)
    image = np.zeros((height,width), np.uint8)
    for x in range(pop.get_chromosomes()[0]._genes.__len__()):
        circle(image, pop.get_chromosomes()[0]._genes[x])
    ##
    m = mse(blank_image, image)
    s = ssim(blank_image, image,multichannel=True)
    # setup the figure
   
    fig = plt.figure('Title')
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(blank_image)
    plt.axis("off")
    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(image)
    plt.axis("off")
    # show the images
#    print('title')
    plt.show()
    
    #increse shape
    pop.increase()
   


'''
if pop.get_chromosomes()[NUMB_OF_ELITE_CHROMOSOMES].get_fitness() < pop.get_chromosomes()[NUMB_OF_ELITE_CHROMOSOMES-1].get_fitness() *2:
        ##
        pop.increase()

'''