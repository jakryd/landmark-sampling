from random import randrange
import math
import random

N = 6
arr = [[0 for x in range(N)] for y in range(N)]
arr[0][1] = 10
arr[0][2] = 15
arr[0][3] = 20
arr[0][4] = 25
arr[0][5] = 30
arr[1][2] = 35
arr[1][3] = 25
arr[1][4] = 45
arr[1][5] = 15
arr[2][3] = 30
arr[2][4] = 10
arr[2][5] = 25
arr[3][4] = 35
arr[3][5] = 20
arr[4][5] = 15

for i in range(N):
    for j in range(N):
        arr[j][i] = arr[i][j]


def swap(list, pos1, pos2):
    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list

def shuffle(list):
    for i in range(100):
        swap(list, randrange(N), randrange(N))
    return list

def cost(list):
    cost = 0

    for i in range(N):
        a = 0
        b = 0
        a = list[i]
        if (i + 1 == N):
            b = list[0]
        else:
            b = list[i + 1]
        cost += arr[a][b]
    return cost

def calc_fitness(list,pop_size):
    fitness = []
    for i in range(pop_size):
        print(list[i], (cost(list[i])))
        fitness.append(1/(cost(list[i]) + 1)) ## TODO 0
    return fitness

def normalize_fitness(fitness, pop_size):
    sum = 0
    for i in range(pop_size):
        sum += fitness[i]
    for i in range(pop_size):
        fitness[i] = fitness[i]/sum
    return fitness

def max_val(list, population, pop_size):
    max = -math.inf
    num = 0
    for i in range(pop_size):
        if list[i] > max:
            max = list[i]
            num = i
    return population[num]

def pick_one(list, prob):
    index = 0
    r = random.uniform(0, 1)

    while(r > 0):
        r = r - prob[index]
        index += 1

    index -= 1
    return list[index]

def mutate(order):
    order = swap(order, randrange(N), randrange(N))
    return order

def nextGeneration(population, fitness, pop_size):
    newPopulation = [[0 for x in range(N)] for y in range(pop_size)]
    best = math.inf
    best_route = []

    for _ in range(5):
        for i in range(pop_size):
            order = pick_one(population, fitness)
            c = cost(order)
            print(order, c)

            if(best > c) :
                best = c
                best_route = order

            order = mutate(order)
            print(order, cost(order))

            newPopulation[i] = order
            print()

        population = newPopulation

    print("BEST", best_route, "cost", best)


def gen_population(list, num):
    population = []
    for i in range(num):
        temp = shuffle(list)[:]
        ##print("temp", temp)
        population.append(temp)
        ##print("pop", population)
    return population

def main():
    cities = [i for i in range(N)]
    print(cities)

    cities = shuffle(cities)
    print("koszt")
    print(cost(cities))
    print()

    pop_size = 10
    population = gen_population(cities, pop_size)
    print("gen pop", population)

    fitness = calc_fitness(population, pop_size)
    print("\nfitness", fitness)
    m = max_val(fitness, population, pop_size)
    print("best", m)
    norm_fitness = normalize_fitness(fitness, pop_size)
    print("fitness in %", norm_fitness)

    print("\ngenerations")
    print(nextGeneration(population, norm_fitness, pop_size))


if __name__ == "__main__":
    main()
