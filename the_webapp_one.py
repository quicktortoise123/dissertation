import streamlit as st
import numpy as np
import random
import pandas as pd
import folium
from streamlit_folium import folium_static
import openrouteservice as ors
import matplotlib.pyplot as plt
import io
pd.set_option('display.max_colwidth',1000)

st.title("WELCOME TO KOLKATA!")
places_name = ["Victoria Memorial",
        "Birla Planetarium",
        "St Paul's Cathedral",
        "ISCON Kolkata",
        "Rabindra Sarovar", 
        "Princep Ghat",
        "St. Johns Church",
        "Biswa Bangla Gate",
        "Tagore House",
        "Marble Palace",
        "Maidan",
        "Eden Garden",
        "Salt Lake Stadium",
        "National Library",
        "Alipore zoo",
        "Gariahat Market",
        "Birla Industrial and Tech Museum",
        "Quest Mall",
        "New Market",
        "Dakshineswar Kali Temple",
        "Park Street",
        "College Street",
        "Science City",
        "Mother House",
        "Kalighat Temple",
        "Netaji Bhavan",
        "Nicco Park",
        "Eco Park",
        "Mother Teresa Wax Museum",
        "Indian Museum"]

places_name=sorted(places_name)
available_time=2
starting_node="Maidan"
available_time=st.number_input('Enter the number of hours', min_value=1.0, value=1.0, step=1.0)
starting_node= st.selectbox("Enter the starting node:", places_name)

total_time_elapsed=0
total_distance_travelled=0

available_time_in_mins = int(available_time) * 60

starting_node_index= places_name.index(starting_node)

df=pd.read_csv(("time.csv"),skiprows=1,skipfooter=1 )
travel_time = df.iloc[:,1:].to_numpy()


df1=pd.read_csv("survey.csv")
visit_time=df1.iloc[:,1:].to_numpy()


df2=pd.read_csv(("distance.csv"),skiprows=1,skipfooter=2 )
distance = df2.iloc[:,1:].to_numpy()

st.write("BASED ON YOUR PREFERENCES THE BEST ROUTES ARE :")


# Global Variables
chromosomes_list = []
visit_time_list = []
travel_time_list = []
distance_covered_list = []
number_of_spots_list = []


def new_pop(n):
  global chromosomes_list
  global visit_time_list
  global travel_time_list
  global distance_covered_list
  global number_of_spots_list
  prev_place_index=0
  next_place_index=0

  chromosomes_list = []
  visit_time_list = []
  travel_time_list = []
  distance_covered_list = []
  number_of_spots_list = []
  while len(chromosomes_list) < n:
    vt=0
    tt=0
    dc=0
    total_time_elapsed = 0

    places_list = []

    vt = int(visit_time[starting_node_index][0])
    total_time_elapsed = total_time_elapsed + vt

    places_list.append(starting_node)
    next_place = starting_node

    while(available_time_in_mins >= total_time_elapsed):
      prev_place = next_place
      prev_place_index=places_name.index(prev_place)
    
      next_place = random.choice(places_name) #randomly choosing new place
      next_place_index = places_name.index(next_place)

      while(next_place in places_list):
        #while next place is not unique --->choose another
        next_place = random.choice(places_name)
        next_place_index = places_name.index(next_place)
      
      if(next_place not in places_list):
        places_list.append(next_place)
        tt = tt + int(travel_time[prev_place_index][next_place_index])
        vt = vt + int(visit_time[next_place_index][0])
        dc = dc + int(distance[prev_place_index][next_place_index])
        total_time_elapsed = total_time_elapsed + int(visit_time[next_place_index][0]) + int(travel_time[prev_place_index][next_place_index])
  
    #deleting the excess element
    if(total_time_elapsed > available_time_in_mins):
      del(places_list[-1])
      tt = tt - int(travel_time[prev_place_index][next_place_index])
      vt = vt - int(visit_time[next_place_index][0])
      dc = dc - int(distance[prev_place_index][next_place_index])
      total_time_elapsed = total_time_elapsed - int(visit_time[next_place_index][0]) - int(travel_time[prev_place_index][next_place_index])

    if places_list not in chromosomes_list:
      chromosomes_list.append( places_list )
      visit_time_list.append(vt)
      travel_time_list.append(tt)
      distance_covered_list.append(dc)
      number_of_spots_list.append(len(places_list))

  X = np.column_stack([distance_covered_list,travel_time_list,visit_time_list,number_of_spots_list  ])

  dist = list(np.array(distance_covered_list))
  #dist=list(-dist)
  tt=list(np.array(travel_time_list))
  #tt=list(-tt)
  vt=np.array(visit_time_list)
  vt=list(vt)
  nspots=np.array(number_of_spots_list)
  nspots=list(nspots)

  X = np.column_stack([dist,tt,vt,nspots])

  return X

def cal_pop_fitness(equation_inputs, pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function caulcuates the sum of products between each input and its corresponding weight.
    fitness = np.sum(pop*equation_inputs, axis=1)
    return fitness

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999

    return parents

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually it is at the center.
    crossover_point = np.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover):
    # Mutation changes a single gene in each offspring randomly.
    for idx in range(offspring_crossover.shape[0]):
        # The random value to be added to the gene.
        random_value = np.random.uniform(-1.0, 1.0, 1)
        offspring_crossover[idx, 3] = offspring_crossover[idx, 3] + random_value
    return offspring_crossover

# Inputs of the equation.
equation_inputs = [-1,-1,1,1]

# Number of the weights we are looking to optimize.
num_weights = 4

sol_per_pop = 30
num_parents_mating = 4

# Defining the population size.
pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
#Creating the initial population.
new_population = new_pop(sol_per_pop)

def normalize(X):
  data = X
  df_without_palces_names = pd.DataFrame(X,columns =['Distance','Travel Time','Visit Time','Spots Visited'])
  normalized_df=df_without_palces_names/df_without_palces_names.max()
  normalized_visit_time_list = np.array(normalized_df['Visit Time'])
  normalized_travel_time_list = np.array(normalized_df['Travel Time'])
  normalized_distance_covered_list = np.array(normalized_df['Distance'])
  normalized_number_of_spots_list = np.array(normalized_df['Spots Visited'])
  return np.array(normalized_df)

num_generations = 500
fitness_list = []
def compareFloatNum(a, b):
  temp_list = []
  for i in range(len(a)):
    if (abs(a[i] - b[i]) < 1e-9):
        temp_list.append(True)
    else:
        temp_list.append(False)
  return temp_list

normalized_pop = normalize(new_population)

for generation in range(num_generations):

    # Measing the fitness of each chromosome in the population.
    fitness = cal_pop_fitness(equation_inputs, normalized_pop)
    
    # Selecting the best parents in the population for mating.
    parents = select_mating_pool(normalized_pop, fitness, 
                                      num_parents_mating)
    
    parents_chromosomes_list = []
    parents_distance_covered_list = []
    parents_travel_time_list = []
    parents_visit_time_list = []
    parents_number_of_spots_list = []

    for i in range(len(parents)):
      for j in range(len(new_population)):
        if np.all( compareFloatNum( parents[i], normalized_pop[j])):
          parents_chromosomes_list.append(chromosomes_list[j])
          parents_distance_covered_list.append(distance_covered_list[j])
          parents_travel_time_list.append(travel_time_list[j])
          parents_visit_time_list.append(visit_time_list[j])
          parents_number_of_spots_list.append(number_of_spots_list[j])
          break;
    


    # Creating the new population based on the parents and offspring.
    normalized_pop[0:parents.shape[0], :] = parents
    X=normalize(new_pop(sol_per_pop - parents.shape[0]))
    
    normalized_pop[parents.shape[0]:, :] = X
    chromosomes_list[:0] = parents_chromosomes_list
    distance_covered_list[:0] = parents_distance_covered_list
    travel_time_list[:0] = parents_travel_time_list
    visit_time_list[:0] = parents_visit_time_list
    number_of_spots_list[:0] = parents_number_of_spots_list
    

    # The best result in the current iteration.
    fitness_result=np.max(np.sum(normalized_pop*equation_inputs, axis=1))
    fitness_list.append(fitness_result)

    
sol_distance_covered_list = parents_distance_covered_list
sol_travel_time_list = parents_travel_time_list
sol_visit_time_list = parents_visit_time_list
sol_number_of_spots_list = parents_number_of_spots_list
best_sol_chromosomes_list = parents_chromosomes_list

#Printing elite 4
data = np.array([best_sol_chromosomes_list, sol_distance_covered_list, sol_travel_time_list, sol_visit_time_list, sol_number_of_spots_list])
df = pd.DataFrame(data.T, columns=['Names','Distance','Travel Time','Visit Time','Spots Visited'])

#dominance among the elites
X = np.column_stack([ sol_distance_covered_list, sol_travel_time_list, sol_visit_time_list, sol_number_of_spots_list])

#print(X)

def dominates(X1, X2):
    if(np.any(X1 < X2) and np.all(X1 <= X2)):
        return True
    else:
        return False

# Distance is minimized
# Travel time is minimized
# Visit time is maximized
# Number of spots is maximized
dist = np.array(sol_distance_covered_list)
tt=np.array(sol_travel_time_list)
vt=np.array(sol_visit_time_list)
vt=-vt
nspots=np.array(sol_number_of_spots_list)
nspots=-nspots

X = np.column_stack([dist,tt,vt,nspots])

dominantX=[]
dom_chromosomes_list=[]
temp=[]

flag = -1

for i in range(len(X)):
  flag=0
  for j in range(len(X)):
   if dominates(X[j], X[i]):
     flag=-1
     break
  if flag == 0:
    dominantX.append(X[i])
    dom_chromosomes_list.append(chromosomes_list[i])

dominantX=np.array(dominantX)
#print(dominantX)

for i in dominantX:
  i[2]=-i[2]
  i[3]=-i[3]



temp1 = []
temp2 = []
for i in range(len(dom_chromosomes_list)):
    if dom_chromosomes_list[i] not in temp1:
      temp1.append(dom_chromosomes_list[i])
      temp2.append(dominantX[i])
dom_chromosomes_list=temp1
dominantX=np.array(temp2)

best_routes=dom_chromosomes_list

dom_distance_covered_list = []
dom_travel_time_list = []
dom_visit_time_list = []
dom_number_of_spots_list = []


for i in dominantX:
  dom_distance_covered_list.append(i[0])
  dom_travel_time_list.append(i[1])
  dom_visit_time_list.append(i[2])
  dom_number_of_spots_list.append(i[3])
  
#Printing the non dominant elites
data = np.array([dom_chromosomes_list, dom_distance_covered_list, dom_travel_time_list, dom_visit_time_list, dom_number_of_spots_list])
df = pd.DataFrame(data.T, columns=['Names','Distance','Travel Time','Visit Time','Spots Visited'])
st.dataframe(df)

non_dominant_optimal_pop = np.column_stack([dom_distance_covered_list,dom_travel_time_list,dom_visit_time_list,dom_number_of_spots_list])
best_match_idx = np.where(abs(fitness - np.max(fitness) ) < 1e-9)

# Getting the best solution after iterating finishing all generations.
#At first, the fitness is calculated for each solution in the final generation.
fitness = cal_pop_fitness(equation_inputs,normalize(non_dominant_optimal_pop))
# Then return the index of that solution corresponding to the best fitness.

best_match_idx = np.where(abs(fitness - np.max(fitness) ) < 1e-9)

best_sol = non_dominant_optimal_pop[best_match_idx, :]
#best_sol = best_sol[0]
best_sol_chromosomes_list = []

for i in range(len(non_dominant_optimal_pop)):
  if np.all(best_sol == non_dominant_optimal_pop[i]):
    best_sol_chromosomes_list.append(dom_chromosomes_list[i])


sol_distance_covered_list = []
sol_travel_time_list = []
sol_visit_time_list = []
sol_number_of_spots_list = []
best_sol = best_sol[0]
for i in best_sol:
  sol_distance_covered_list.append(i[0])
  sol_travel_time_list.append(i[1])
  sol_visit_time_list.append(i[2])
  sol_number_of_spots_list.append(i[3])

st.write("WE RECOMMEND THE FOLLOWING ROUTE : ")
#Recommended route
data = np.array([best_sol_chromosomes_list, sol_distance_covered_list, sol_travel_time_list, sol_visit_time_list, sol_number_of_spots_list])
df = pd.DataFrame(data.T, columns=['Names','Distance','Travel Time','Visit Time','Spots Visited'])
st.dataframe(df)

recommended_route=best_sol_chromosomes_list

data = pd.read_csv("coord.csv")
lat = list(data["LAT"])
lon = list(data["LONG"])
location = list(data["LOCATION"])

fg=folium.FeatureGroup(name="My Map")

for lt, ln, loc in zip(lat, lon, location):
    fg.add_child(folium.Marker(location=[lt, ln], popup= loc, icon=folium.Icon(color='darkblue', icon='thumb-tack', prefix='fa')))

client = ors.Client(key='5b3ce3597851110001cf62482ebf88c9257446b39ecc1e2d47b032df')

map = folium.Map(location=(22.55443963, 88.35131118), zoom_start=15, tiles="Open Street Map")
map.add_child(fg)

fg0=folium.FeatureGroup(name="Recommended Route")

points00 = recommended_route[0]

for i in range(len(points00)):
  j=points00[i]
  fg0.add_child(folium.Marker(location=[lat[location.index(j)], lon[location.index(j)]], popup= j , icon=folium.Icon(color='darkblue', icon='thumb-tack', prefix='fa')))
  j=""

for i in range(len(points00)):
  j=points00[i]
  k=points00[i-1]
  coordinates=[[lon[location.index(j)], lat[location.index(j)]] , [lon[location.index(k)], lat[location.index(k)]]] 
  route = client.directions(
    coordinates=coordinates,
    profile='foot-walking',
    format='geojson',
    options={"avoid_features": ["steps"]},
    validate=False,
  )
  folium.PolyLine(locations=[list(reversed(coord)) 
                           for coord in 
                           route['features'][0]['geometry']['coordinates']],color="black").add_to(fg0)
                          
map.add_child(fg0)

for i in range(len(best_routes)):

  fg1=folium.FeatureGroup(name="Route "+str(i))

  points = best_routes[i]

  for i in range(len(points)):
    j=points[i]
    fg1.add_child(folium.Marker(location=[lat[location.index(j)], lon[location.index(j)]], popup= j , icon=folium.Icon(color='darkblue', icon='thumb-tack', prefix='fa')))
    j=""

  for i in range(len(points)):
    j=points[i]
    k=points[i-1]
    coordinates=[[lon[location.index(j)], lat[location.index(j)]] , [lon[location.index(k)], lat[location.index(k)]]] 
    route = client.directions(
      coordinates=coordinates,
      profile='foot-walking',
      format='geojson',
      options={"avoid_features": ["steps"]},
      validate=False,
    )
    folium.PolyLine(locations=[list(reversed(coord)) 
                            for coord in 
                            route['features'][0]['geometry']['coordinates']],color="red").add_to(fg1)
                          
  map.add_child(fg1)

map.add_child(folium.LayerControl())

folium_static(map)
map.save("Map.html")