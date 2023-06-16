#!/usr/bin/env python
# coding: utf-8

# # **Computational Intelligence project_B 2023**

# In[340]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math


from sklearn.preprocessing import MinMaxScaler
from numpy.linalg import norm


from deap import base
from deap import creator 
from deap import tools




# ## **Data Preparation**

# ### **Reading csv file and creating dataset**

# In[475]:


path = r"C:\Users\Trinity\Documents\GA_Project\dataset-HAR-PUC-Rio.csv" 
dataset= pd.read_csv(path, delimiter=";", decimal = ",", low_memory=False) # Read the file
data = dataset.drop(["user","gender","age","how_tall_in_meters","weight","body_mass_index","Class"], axis=1)
label = dataset["Class"]
dataset


# In[476]:


scaler = MinMaxScaler(feature_range=(0,1))

scaled_data = scaler.fit_transform(data)


# ### **Mean vectors of every class**

# In[477]:


scaled_data = pd.DataFrame(scaled_data)
scaled_data["Class"] = label
mean_vectors = scaled_data.groupby('Class').mean()
mean_vectors


# ## **Genetic Algorithm**

# In[478]:


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def attr():
    return random.random()


toolbox.register("attr_real", attr)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_real, 12)
toolbox.register("population", tools.initRepeat, list, toolbox.individual) 

#EVALUATION FUNCTION
def eval(individual, vecs, c):
    cosines = []
    for m in mean_vectors.to_numpy():
        cos = np.dot(m,individual)/(norm(m)*norm(individual))
        cosines.append(cos)

    f = (cosines[0] + c*(1-(1/4)*(sum(cosines[1:4]))))/(1+c)
    return f,



#Genetic operators
toolbox.register("evaluate", eval, vecs=mean_vectors, c=0.1)
toolbox.register("mate", tools.cxBlend, alpha = 0.1)
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=1.0, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=4)
toolbox.register("selectBest", tools.selBest, k=1, fit_attr='fitness')




# ### **Evolution**

# In[479]:


avg_max = []
best_gens = []
best_score = 0
for i in range(10):
     
    #Population creation
    pop = toolbox.population(n=200)
    pop

    #Evalutation
    fitnesses = list(map(toolbox.evaluate, pop))

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    

    # CXPB = crossover probability 
    # MUTPB = mutation probability
    CXPB, MUTPB = 0.6, 0.1

    fits = [ind.fitness.values[0] for ind in pop]
    
    #Find best score and average of all generations
    if max(fits) > best_score:
        best_score = max(fits)
        
    best_avg = sum(fits) / len(pop)

    #Find elit individualin this generation
    for ind, fit in zip(pop, fits):
        if fit == max(fits):
            elit = ind 
            
    #Find elit individual in all generations       
    for ind, fit in zip(pop, fits):
        if fit == best_score:
            b_elit = ind

    #Generations
    g = 0
    best_gen = g
    temp_max = []
    
    mean = sum(fits) / len(pop)
    print("-- Generation %i --" % g)
    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)
    print("  Elit individual ", elit)
    
    
    while max(fits) < 1 and g < 100:
        
     
        temp_max.append(max(fits))
        g = g + 1
        print("-- Generation %i --" % g)
        
        #Select next generation individuals
        offsprings = toolbox.select(pop, len(pop))
        
        #Clone selected individuals
        offsprings = list(map(toolbox.clone, offsprings)) 

        #Crossover and mutation on offsprings
        for child1, child2 in zip(offsprings[::2], offsprings[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
    

        for mutant in offsprings:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
            
    
        #Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offsprings if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
                
        # Replace old population with offsprings
        pop[:] = offsprings
    

        fits = [ind.fitness.values[0] for ind in pop]
        
    
        #Find elit individual in this generation
        for ind, fit in zip(pop, fits):
            if fit == max(fits):
                elit = ind    
   
        if max(fits) > best_score:
            best_score = max(fits)
            
        #Find general elit endividual     
        for ind, fit in zip(pop, fits):
            if fit == best_score:
                b_elit = ind
        
        #Generation of best average score
        temp_avg = sum(fits) / len(pop)
        if temp_avg > best_avg:
            best_avg = temp_avg
            best_gen = g
    
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
    
        """
        req = mean + mean/100000
    
        if n_mean < req:
        evolution = False
        else :
        mean = n_mean
        """ 
   

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        print("  Elit individual ", elit)

    avg_max.append(temp_max)
    best_gens.append(best_gen)
    
    



    


# ### **Average evolution**

# In[480]:


avg_max = np.array(avg_max)
avg_max = np.transpose(avg_max)
mean_max = np.mean(avg_max, axis=1)
mean_max


# ## **Evolution plot**

# In[482]:


plt.title("Evolution: pop_size=200, CXPB=0.6, MUTPB =0.1")
plt.plot(list(i for i in range(100)),mean_max)


# In[483]:


print("Best individual: ",b_elit)
print("Best score : %s " % best_score)
print("Best average score : %s" % best_avg, " on generation: ",best_gen)



