# -*- coding: utf-8 -*-

import numpy
import random
from textProcessor import textProcessor
from Stemmer import Stemmer
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

parser = argparse.ArgumentParser(description='File path parser')
parser.add_argument('--source', '-s', type=str, required=True, help='Path to the file or folder with files')
parser.add_argument('--dest', '-d', type=str, help='Path to the results saving folder')
parser.add_argument('--freq', '-f', default=1, type=int, help='Если слово встречактся в документе больше раз, чем'
                                                              ' задано здесь, то оно подлежит удалению')

args = parser.parse_args()

words = []

source = args.source
dest = args.dest
freq = args.freq

# if str(source.find('.txt')) == -1:
if source[-3:len(source)] != 'txt':
    for k in range(1, 7):
        f = open(source +'\T' + str(k) + '.txt')
        # f = open('mydoc\\T'+str(i)+'.txt')
        words.append(f.read())
else:
    f = open(source)
    # f = open('text2.txt')
    for line in f.readlines():
        words.append(line.strip())

TP = textProcessor()
ST = Stemmer()

words = [w.lower() for w in words]
words = TP.remove_symbols(words)
words = TP.remove_stopwords(words)
print(words)

stemmed = []
for sentence in words:
    s = [ST.stem(i) for i in sentence]                # производится стемминг
    stemmed.append(s)
print(stemmed)

keys = TP.remove_unique(stemmed, freq)                # удаление слов, встречающихся во всех документах более freq раз
print(keys)
print(len(keys))

table, disp_table = TP.table_generator(keys, stemmed)
print(disp_table)

# Сингулярное разложение
LA = numpy.linalg
freqMatrix = numpy.array(table)
print(freqMatrix)
terms, s, docs = LA.svd(freqMatrix, full_matrices=False)
assert numpy.allclose(freqMatrix, numpy.dot(terms, numpy.dot(numpy.diag(s), docs)))
print(terms)
print('          ')
print(s)
print('          ')
print(docs)
print('          ')
s[2:] = 0
new_a = numpy.dot(terms, numpy.dot(numpy.diag(s), docs))    # Двумерное сингулярное разложение
print(new_a)

# Построение графика
matplotlib.rc('font', family='Arial', size='16')
fig = plt.figure()
axes = Axes3D(fig)


for k in range(len(freqMatrix[0])):
    shift = random.uniform(-0.2, 0.3)
    axes.scatter(docs[0][k], docs[1][k], docs[2][k], color='b')   # scatter - метод для нанесения маркера в точке
    axes.plot([docs[0][k], docs[0][k]], [docs[1][k], docs[1][k]], zs=[docs[2][k], 0], color='k', dashes=[8, 4, 2, 4, 2, 4])
    axes.text(docs[0][k], docs[1][k], docs[2][k] + shift, str(k + 1))

for j in range(len(freqMatrix)):
    shift = random.uniform(-0.2, 0.3)
    axes.scatter(terms[j][0], terms[j][1], terms[j][2], color='r')
    axes.plot([terms[j][0], terms[j][0]], [terms[j][1], terms[j][1]], zs=[terms[j][2], 0], color='k', dashes=[8, 4, 2, 4, 2, 4])
    axes.text(terms[j][0], terms[j][1], terms[j][2] + shift, str(keys[j]))

if dest is not None:
    print(dest)
    plt.savefig(dest+'\graphic', fmt='svg')
    with open(dest+r'\results.txt', 'w+') as f:
        k = 0
        j = 0
        f.write('Матрица термы-на-документы\n')
        for line in disp_table:
            f.write(str(line[::-1])+'\n')
        f.write('Координаты термов\n')
        for line in terms:
            f.write(str(line[:3]) + keys[k] + '\n')
            k += 1
        f.write('Координаты документов\n')
        for line in docs.transpose():             # Матрица с коорд. документов транспонирована для лучшего вывода
            f.write(str(line[:3])+'Doc#'+str(j+1)+'\n')
            j += 1
    f.close()

# Расчитаем расстояния до всех термов от каждого документа

termCords = [line[:3] for line in terms]
docCords = [line[:3] for line in docs.transpose()]

statistics = {}
index = 0
for doc in docCords:
    k = 0
    distDictionary = {}
    for term in termCords:
        dist = numpy.sqrt(((float(term[0])-float(doc[0]))**2)+((float(term[1])-float(doc[1]))**2)+((float(term[2])-float(doc[2]))**2))
        distDictionary[keys[k]] = dist
        k += 1
    statistics[index+1] = distDictionary
    index += 1

l = lambda x: x[1]
print(sorted(statistics[1].items(), key=l, reverse=True))

# print(statistics)
plt.show()
