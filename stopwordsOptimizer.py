# -*- coding: utf-8 -*-

import re

# stopwords = []
# f = open(u'стоп-слова.txt', "r", -1, "utf-8")
# for line in f.readlines():
#     stopwords.append(line)
# f.close()
#
# stopwords.sort()
# print(stopwords)
#
# v = open(u'oldwords.txt', "w", -1, "utf-8")
# for w in stopwords:
#     v.write(str(w))

stopwords = u'''1 2 3 4 5 6 7 8 9 0 а анализ алгоритм без более больше больший большаяя большее большие будет будут будешь буде будем будто бы был была были было быть в вам вас вдруг ведь вектор вес вид во вон вот впрочем вид вода выполнение выполнить выполнять все всегда всего всех всю вы где говорил должны добавить должен дополнительно да даже данный данная данной два для до другой его ее ей ему если есть еще ж же жизнь за зачем здесь и из или им иногда использовать использованный использованная их к кажется как какая какой когда конечно который кто куда ли лучше между матрица матрицы матрицу матрице меня метод мне много может можно мой моя мы на над надо наконец нас начало начала нахождение наш не него нее ней нельзя некоторый нет некоторый некоторая некоторые необходимо ни нибудь никогда ним них ничего но ну о об один он она отдельный отдельная отдельное оценка очередь осуществляет они опять от перед по под после потом потому почти при про раз разве результат результата результатов результатом с следует следовать сам свою себе соответствует соответствующий соответствующая соответствующие себя сегодня сейчас сказал сказала сказать следующий следующих со совсем так так такой там тебя тем теперь то тогда того тоже только том тот три тут ты у уж уже хорошо хоть характеристика формула функции функция форма формы удаление чего человек чем через что чтоб чтобы чуть эта эти этих этого этой этом этот эту является являться этап элемент я a ak bk c i k m n r rm s sk u uk v vk x y'''

stopwords.strip()
print(stopwords)

fromfile = []
f = open(u'oldwords.txt', "r", -1, "utf-8")
for line in f.readlines():
    fromfile.append(line.strip())
f.close()

stopwordsList = stopwords.split(' ')

all = stopwordsList+fromfile
for w in all:
    w.strip()

allStopwords = set(all)

print(allStopwords)

finalList = list(allStopwords)
finalList.sort()
print(finalList)

v = open(u'stopwords.txt', "w", -1, "utf-8")
for w in finalList:
    v.write(str(w+'\n'))
