import numpy as np
from numpy.random import randint
from random import random as rnd
from random import gauss, randrange
import pop as pop


# Создание первоначальной популяции потенциальных решений, гены которых генерируются случайным образом.
def individual(number_of_genes, upper_limit, lower_limit):
    individuals = [round(rnd() * (upper_limit - lower_limit)
                         + lower_limit, 1) for x in range(number_of_genes)]
    return individuals


# Функция принимает количество генов, верхний и нижний пределы для генов в качестве входных данных и создает
# индивидуума.
def population(number_of_individuals,
               number_of_genes, upper_limit, lower_limit):
    return [individual(number_of_genes, upper_limit, lower_limit)
            for x in range(number_of_individuals)]


# Функция расчета пригодности определяет значение пригодности индивидуума.
def fitness_calculation(individuals):
    fitness_value = sum(individuals)
    return fitness_value


# Функция выбора колеса рулетки принимает кумулятивные суммы и случайно сгенерированное значение для процесса выбора
# и возвращает номер выбранного человека
def roulette(cum_sum, chance):
    variable = list(cum_sum.copy())
    variable.append(chance)
    variable = sorted(variable)
    return variable.index(chance)


# Функция выбора (случайная фигура)
def selection(generation, method='Fittest Half'):
    generation['Normalized Fitness'] = \
        sorted([generation['Fitness'][x] / sum(generation['Fitness'])
                for x in range(len(generation['Fitness']))], reverse=True)
    generation['Cumulative Sum'] = np.array(
        generation['Normalized Fitness']).cumsum()
    if method == 'Roulette Wheel':
        selected = []
        for x in range(len(generation['Individuals']) // 2):
            selected.append(roulette(generation
                                     ['Cumulative Sum'], rnd()))
            while len(set(selected)) != len(selected):
                selected[x] = \
                    (roulette(generation['Cumulative Sum'], rnd()))
        selected = {'Individuals':
                        [generation['Individuals'][int(selected[x])]
                         for x in range(len(generation['Individuals']) // 2)]
            , 'Fitness': [generation['Fitness'][int(selected[x])]
                          for x in range(
                    len(generation['Individuals']) // 2)]}
    elif method == 'Fittest Half':
        selected_individuals = [generation['Individuals'][-x - 1]
                                for x in range(int(len(generation['Individuals']) // 2))]
        selected_finesses = [generation['Fitness'][-x - 1]
                             for x in range(int(len(generation['Individuals']) // 2))]
        selected = {'Individuals': selected_individuals,
                    'Fitness': selected_finesses}
    elif method == 'Random':
        selected_individuals = \
            [generation['Individuals']
             [randint(1, len(generation['Fitness']))]
             for x in range(int(len(generation['Individuals']) // 2))]
        selected_finesses = [generation['Fitness'][-x - 1]
                             for x in range(int(len(generation['Individuals']) // 2))]
        selected = {'Individuals': selected_individuals,
                    'Fitness': selected_finesses}
    return selected


# Функция сопряжения (взвешенное случайное спаривание)
def pairing(elite, selected, method='Fittest'):
    individuals = [elite['Individuals']] + selected['Individuals']
    fitness = [elite['Fitness']] + selected['Fitness']
    if method == 'Fittest':
        parents = [[individuals[x], individuals[x + 1]]
                   for x in range(len(individuals) // 2)]
    if method == 'Random':
        parents = []
        for x in range(len(individuals) // 2):
            parents.append(
                [individuals[randint(0, (len(individuals) - 1))],
                 individuals[randint(0, (len(individuals) - 1))]])
            while parents[x][0] == parents[x][1]:
                parents[x][1] = individuals[
                    randint(0, (len(individuals) - 1))]
    if method == 'Weighted Random':
        normalized_fitness = sorted(
            [fitness[x] / sum(fitness)
             for x in range(len(individuals) // 2)], reverse=True)
        cumulative_sum = np.array(normalized_fitness).cumsum()
        parents = []
        for x in range(len(individuals) // 2):
            parents.append(
                [individuals[roulette(cumulative_sum, rnd())],
                 individuals[roulette(cumulative_sum, rnd())]])
            while parents[x][0] == parents[x][1]:
                parents[x][1] = individuals[
                    roulette(cumulative_sum, rnd())]
    return parents


# Функция спаривания (выбор 2-ух точек)
def mating(parents, method='Single Point'):
    if method == 'Single Point':
        pivot_point = randint(1, len(parents[0]))
        offsprings = [parents[0] \
                          [0:pivot_point] + parents[1][pivot_point:], parents[1]
                      [0:pivot_point] + parents[0][pivot_point:]]
    if method == 'Two Pionts':
        pivot_point_1 = randint(1, len(parents[0] - 1))
        pivot_point_2 = randint(1, len(parents[0]))
        while pivot_point_2 < pivot_point_1:
            pivot_point_2 = randint(1, len(parents[0]))
        offsprings = [parents[0][0:pivot_point_1] +
                      parents[1][pivot_point_1:pivot_point_2] +
                      [parents[0][pivot_point_2:]], [parents[1][0:pivot_point_1] +
                                                     parents[0][pivot_point_1:pivot_point_2] +
                                                     [parents[1][pivot_point_2:]]]]
    return offsprings


# Функция мутации (сброс мутационной фигуры)
def mutation(individuals, upper_limit, lower_limit, mutation_rate=2,
             method='Reset', standard_deviation=0.001):
    gene = [randint(0, 7)]
    for x in range(mutation_rate - 1):
        gene.append(randint(0, 7))
        while len(set(gene)) < len(gene):
            gene[x] = randint(0, 7)
    mutated_individual = individuals.copy()
    if method == 'Gauss':
        for x in range(mutation_rate):
            mutated_individual[x] = \
                round(individuals[x] + gauss(0, standard_deviation), 1)
    if method == 'Reset':
        for x in range(mutation_rate):
            mutated_individual[x] = round(rnd() * \
                                          (upper_limit - lower_limit) + lower_limit, 1)
    return mutated_individual


# Следующее поколение создано с использованием генетических операций. Элитарность может быть введена в генетический
# алгоритм при создании следующего поколения
def next_generation(gene, upper_limit, lower_limit):
    elite = {}
    next_gen = {}
    elite['Individuals'] = gene['Individuals'].pop(-1)
    elite['Fitness'] = gene['Fitness'].pop(-1)
    selected = selection(gene)
    parents = pairing(elite, selected)
    offsprings = [[[mating(parents[x])
                    for x in range(len(parents))]
                   [y][z] for z in range(2)]
                  for y in range(len(parents))]
    offsprings1 = [offsprings[x][0]
                   for x in range(len(parents))]
    offsprings2 = [offsprings[x][1]
                   for x in range(len(parents))]
    unmutated = selected['Individuals'] + offsprings1 + offsprings2
    mutated = [mutation(unmutated[x], upper_limit, lower_limit)
               for x in range(len(gene['Individuals']))]
    unsorted_individuals = mutated + [elite['Individuals']]
    unsorted_next_gen = \
        [fitness_calculation(mutated[x])
         for x in range(len(mutated))]
    unsorted_fitness = [unsorted_next_gen[x]
                        for x in range(len(gene['Fitness']))] + [elite['Fitness']]
    sorted_next_gen = \
        sorted([[unsorted_individuals[x], unsorted_fitness[x]]
                for x in range(len(unsorted_individuals))],
               key=lambda x: x[1])
    next_gen['Individuals'] = [sorted_next_gen[x][0]
                               for x in range(len(sorted_next_gen))]
    next_gen['Fitness'] = [sorted_next_gen[x][1]
                           for x in range(len(sorted_next_gen))]
    gene['Individuals'].append(elite['Individuals'])
    gene['Fitness'].append(elite['Fitness'])
    return next_gen


# Проверяем изменились ли максимальные значения пригодности
def fitness_similarity_check(max_fitness, number_of_similarity):
    result = False
    similarity = 0
    for n in range(len(max_fitness) - 1):
        if max_fitness[n] == max_fitness[n + 1]:
            similarity += 1
        else:
            similarity = 0
    if similarity == number_of_similarity - 1:
        result = True
    return result


Result_file = 'GA_Results.txt'


# Запуск генетического алгоритма для 20 человек в каждом поколении
def first_generation(pop):
    fitness = [fitness_calculation(pop[x])
               for x in range(list(pop))]
    sorted_fitness = sorted([[pop[x], fitness[x]]
                             for x in range(len(pop))], key=lambda x: x[1])
    populations = [sorted_fitness[x][0]
                   for x in range(len(sorted_fitness))]
    fitness = [sorted_fitness[x][1]
               for x in range(len(sorted_fitness))]
    return {'Individuals': populations, 'Fitness': sorted(fitness)}


gen = [first_generation(pop)]
fitness_avg = np.array([sum(gen[0]['Fitness']) /
                        len(gen[0]['Fitness'])])
fitness_max = np.array([max(gen[0]['Fitness'])])
res = open(Result_file, 'a')
res.write('\n' + str(gen) + '\n')
res.close()
finish = False
while not finish:
    if max(fitness_max) > 6:
        break
    if max(fitness_avg) > 5:
        break
    if fitness_similarity_check(fitness_max, 50):
        break
    gen.append(next_generation(gen[-1], 1, 0))
    fitness_avg = np.append(fitness_avg, sum(
        gen[-1]['Fitness']) / len(gen[-1]['Fitness']))
    fitness_max = np.append(fitness_max, max(gen[-1]['Fitness']))
    res = open(Result_file, 'a')
    res.write('\n' + str(gen[-1]) + '\n')
    res.close()
