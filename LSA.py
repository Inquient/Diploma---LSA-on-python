# -*- coding: utf-8 -*-

import numpy
import argparse
from textProcessor import textProcessor
from Stemmer import Stemmer
import random
from FileCryptor import FileCryptor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser(description='File path parser')
parser.add_argument('--source', '-s', type=str, required=True, help='Путь к файлу или папке с документами')
parser.add_argument('--dest', '-d', type=str, help='Место сохранения итоговых данных')
parser.add_argument('--freq', '-f', default=1, type=int, help='Если слово встречактся в документе больше раз, чем'
                                                              ' задано здесь, то оно подлежит удалению')
parser.add_argument('--pca', action='store_true', help='Если активировать этот ключ, то будет срабатывать'
                                                            'анализ главных компонент перед отбором признаков')


def plotGraphic(docs, terms, keys, pca=False):
    fig = plt.figure()
    axes = Axes3D(fig)

    if(pca == False):
        docs = docs.transpose()
        # x = numpy.arange(-1, 1, 0.1)
        # y = numpy.arange(-1, 1, 0.1)
        # xgrid, ygrid = numpy.meshgrid(x, y)
        # axes.plot_surface(xgrid, ygrid, 0, color='b', alpha='0.33')

    i = 1
    for doc in docs:
        axes.scatter(doc[0], doc[1], doc[2], color='b', edgecolor='k')
        axes.plot([doc[0], doc[0]], [doc[1], doc[1]], zs=[doc[2], 0], color='k',
                  dashes=[8, 4, 2, 4, 2, 4])
        axes.text(doc[0], doc[1], doc[2], str(i))
        i += 1

    j = 0
    for term in terms:
        axes.scatter(term[0], term[1], term[2], color='r', edgecolor='k')
        axes.plot([term[0], term[0]], [term[1], term[1]], zs=[term[2], 0], color='k',
                  dashes=[8, 4, 2, 4, 2, 4])
        shift = random.uniform(0.05, 0.1)
        axes.text(term[0], term[1], term[2]+shift, str(keys[j]))
        j += 1


args = parser.parse_args()

words = []

source = args.source
dest = args.dest
freq = args.freq
pca = args.pca

# if str(source.find('.txt')) == -1:
if source[-3:len(source)] != 'txt':
    for k in range(1, 8):
        f = open(source +'/Doc' + str(k) + '.txt',encoding='utf-8')
        # f = open('mydoc\\T'+str(i)+'.txt')
        words.append(f.read())
else:
    f = open(source,encoding='utf-8')
    # f = open('text2.txt')
    for line in f.readlines():
        words.append(line.strip())

TP = textProcessor()
ST = Stemmer()

words = [w.lower() for w in words]      # Переводим все строки в нижний регистр
words = TP.remove_symbols(words)        # Удаляем стоп-символы
words = TP.remove_stopwords(words)      # Удаляем стоп-слова
# print(words)

stemmed = []
for sentence in words:
    s = [ST.stem(i) for i in sentence]                # Производится стемминг
    stemmed.append(s)
# print(stemmed)

keys = TP.remove_unique(stemmed, freq)                # Удаление слов, встречающихся во всех документах более freq раз/
# print(keys)                                           # По умолчанию частота freq=1 равна еденице
# print(len(keys))                                      # Получаем массив ключевых слов - термов

table, disp_table = TP.table_generator(keys, stemmed)       # Формируем частотную матрицу - table
# print(disp_table)                                           # И таблицу частоты  встречаемости - disp_table

# Сингулярное разложение
LA = numpy.linalg
freqMatrix = numpy.array(table)
# print(freqMatrix)
terms, s, docs = LA.svd(freqMatrix, full_matrices=False)
assert numpy.allclose(freqMatrix, numpy.dot(terms, numpy.dot(numpy.diag(s), docs)))
# s[2:] = 0
new_a = numpy.dot(terms, numpy.dot(numpy.diag(s), docs))

# Вывод графика и Анализ главных компонент, если требуется
if(pca):
    pca = PCA(n_components=3)
    fit_docs = pca.fit_transform(docs)
    fit_terms = pca.fit_transform(terms)
    print(fit_terms)
    print('          ')
    print(fit_docs)
    print('          ')
    plotGraphic(fit_docs, fit_terms, keys, pca=True)
else:
    print(terms)
    print('          ')
    print(s)
    print('          ')
    print(docs)
    print('          ')
    plotGraphic(docs, terms, keys)


if(pca):
    termCords = fit_terms                                 # Для разложения с анализом главных компонент
    docCords = fit_docs
else:
    termCords = [line[:3] for line in terms]                # Для обычного разложения
    docCords = [line[:3] for line in docs.transpose()]


# Расчитаем расстояния до всех термов от каждого документа
                # Результаты расчётов поместим в словарь словарей statistics
statistics = {} # В нём по номеру документа хранятся словари из пар- (терм: расстояние от терма до данного документа)
index = 0       # Получим наглядный словарь по расстояниям от каждого терма до каждого документа
for doc in docCords:
    k = 0
    distDictionary = {}
    for term in termCords:
        dist = numpy.sqrt(((float(term[0])-float(doc[0]))**2)+((float(term[1])-float(doc[1]))**2)+((float(term[2])-float(doc[2]))**2))
        distDictionary[keys[k]] = dist
        k += 1
    statistics[index+1] = distDictionary
    index += 1


                    # Теперь отсортируем каждый словарь из statistics по возрастанию расстояния от документа до терма
l = lambda x: -x[1] # Таким образом получим упорядоченные списки, где первые термы больше всего соответсвуют
index = 1           # данному документу.
www = []
while index <= statistics.__len__():
    end = sorted(statistics[index].items(), key=l, reverse=True)
    print(index, end[:5])      # Из всех значений термов для каждого документа оставим 5 наиболее длизких по расстоянию
    end_word_list = dict(end[:5])   # И выведем их
    www.append(list(end_word_list.keys()))
    index += 1


sss = set()             # Теперь составим множество всех термов, которые находятся по отношению к своим документам в
for list in www:        # пятёрке наиболее близких и выведем их
    for item in list:
        sss.add(item)
print(sss)              # Это и будут термы, наиболее точно передающие тему и смысл всего набора документов, т.е. текста

# Выводим информацию об анализе в файл
if dest is not None:
    print(dest)
    plt.savefig(dest+'/graphic', fmt='png')
    with open(dest+r'/results.txt', 'w+', encoding='utf-8') as f:
        k = 0
        j = 0
        f.write('Матрица термы-на-документы\n')
        for line in disp_table:
            f.write(str(line[::-1])+'\n')

        if(pca):
            f.write('Координаты термов\n')
            for line in fit_terms:
                f.write(str(line[:3]) + keys[k] + '\n')
                k += 1
            f.write('Координаты документов\n')
            for line in fit_docs.transpose():  # Матрица с коорд. документов транспонирована для лучшего вывода
                f.write(str(line[:3]) + 'Doc#' + str(j + 1) + '\n')
                j += 1
        else:
            f.write('Координаты термов\n')
            for line in terms:
                f.write(str(line[:3]) + keys[k] + '\n')
                k += 1
            f.write('Координаты документов\n')
            for line in docs.transpose():             # Матрица с коорд. документов транспонирована для лучшего вывода
                f.write(str(line[:3])+'Doc#'+str(j+1)+'\n')
                j += 1
        f.write(str(sss))
    f.close()

plt.show()