# -*- coding: utf-8 -*-
import numpy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Stemmer import *

stop_words = u'''и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот
             от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь
             опять уж вам сказал ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их
             чем была сам чтоб без будто человек чего раз тоже себе под жизнь будет ж тогда кто этот говорил
             того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее кажется сейчас были
             куда зачем сказать всех никогда сегодня можно при наконец два об другой хоть после над больше тот
             через эти нас про всего них какая много разве сказала три эту моя впрочем хорошо свою этой перед
             иногда лучше чуть том нельзя такой им более всегда конечно всю между так вон'''

words = [u'Британская полиция знает о местонахождении основателя WikiLeaks',
         u'В суде США начинается процесс против россиянина, рассылавшего спам',
         u'Церемонию вручения Нобелевской премии мира бойкотируют 19 стран',
         u'В Великобритании арестован основатель сайта WikiLeaks Джулиан Ассандж',
         u'Украина игнорирует церемонию вручения Нобелевской премии',
         u'Шведский суд отказался рассматривать апелляцию основателя WikiLeaks',
         u'НАТО и США разработали планы обороны стран Балтии против России',
         u'Полиция Великобритании нашла основателя WikiLeaks, но, не арестовала',
         u'В Стокгольме и Осло сегодня состоится вручение Нобелевских премий']

# words = [u'В ДНР жалуются: Россия не дает денег на пенсию',
#          u'Крушение поезда в Индии: число погибших превысило 100 человек',
#          u'ДНР решила создать компьютерные игры о боях за Дебальцево и аэропорт',
#          u'Турция может не пойти в ЕС, а вступить в ШОС',
#          u'На Донбассе задержан минометчик ДНР, который обстреливал Майорск',
#          u'Политолог рассказал, как ускорить деоккупацию Крыма и Донбасса',
#          u'Столтенбкрг обсудил с Трампом будущее НАТО',
#          u'Обама призвал дать Трампу время и не ждать худшего',
#          u'Под Киевом столкнулись грузовик и автобус, есть погибший']

stop_words_list = stop_words.split(" ")
_stemmer = Stemmer()

words = [w.lower() for w in words]                          # приводим все строки к нижнему регистру

unsymboled = []
for word in words:
    s = re.sub(r'[.,!?;:{}[]()-_]', '', word)         # с помощью регулярных выражений удаляем знаки препинания
    unsymboled.append(s)

listed = [s.split(" ") for s in unsymboled]                 # разделяем предложения на отдельные слова

new = []
for sentence in listed:
    s = [i for i in sentence if i not in stop_words_list]   # удаляем стоп-символы
    new.append(s)

result = []
for sentence in new:
    s = [_stemmer.stem(i) for i in sentence]                # производится стемминг
    result.append(s)

print(result)

# преобразование массива result в строку для удаления уникальных вхождений
text = [" ".join(i) for i in result]
text = " ".join(text)
words = text.split(" ")

#print(words)                                                # Все слова по отдельности
#print(text)                                                 # Сам текст

newtext = ''
for word in words:
    i = text.count(word)
    if i > 1:
        newtext += word
        newtext += ' '

m = set(newtext.split(" "))
m.remove('')

l = [i for i in m]
l.sort()                                # Отсортированые слова, встречающиеся в тексте более 1-го раза
print(l)

# for fragment in result:               # Та же генерация таблици, но возвращает столбцы
#     col = [fragment, []]
#     for w in l:
#         amount = fragment.count(w)
#         col[1].append(amount)
#     print(col)
#     table.append(col)

table = []                              # Создаём таблицу для вывода частоты встречаемости слов
disp_table = []                         # в каждом фрагменте текста

for w in l:
    disp_row = [w, []]                  # Строка таблицы - список из самого слова и списка количества его вхождений в каждом фрагменте
    row = []
    for fragment in result:
        amount = fragment.count(w)      # Считаем число вхождений в текущем фрагменте
        disp_row[1].append(amount)      # Посчитав заносим их в список
        row.append(amount)
    # print(disp_row)
    disp_table.append(disp_row)
    table.append(row)                   # Вносим строки в таблицу

# print(disp_table)
# print(table)

#Сингулярное разложение
LA = numpy.linalg
a = numpy.array(table)
print(a)
U, s, Vh = LA.svd(a, full_matrices=False)
assert numpy.allclose(a, numpy.dot(U, numpy.dot(numpy.diag(s), Vh)))
# print(U)
# print('          ')
# print(s)
# print('          ')
# print(Vh)
# print('          ')
s[2:] = 0
new_a = numpy.dot(U, numpy.dot(numpy.diag(s), Vh)) # Двумерное сингулярное разложение
# print(new_a)

# Пример 1.4.1

matplotlib.rc('font', family='Arial')
fig = plt.figure()
axes = Axes3D(fig)
# arrowprops = {
#         'arrowstyle': '->',
#     }
# # x1 = 0.05
# # y1 = 0.05
for i in range(len(a[0])):
    axes.scatter(Vh[0][i], Vh[1][i], Vh[2][i], color='b')   # scatter - метод для нанесения маркера в точке (1.0, 1.0)
    axes.text(Vh[0][i], Vh[1][i], Vh[2][i], str(i+1))
#     plt.annotate(str(i+1), xy=(Vh[0][i], Vh[1][i]), xytext=(Vh[0][i] + x1, Vh[1][i] + y1), arrowprops=arrowprops)
#     y1+=0.05
#
# x2 = 0.05
# y2 = 0.05
for j in range(len(a)):
    axes.scatter(U[j][0], U[j][1], U[j][2], color='r')
    axes.text(U[j][0], U[j][1], U[j][2], str(l[j]))
    # plt.annotate(str(l[j]), xy=(U[j][0], U[j][1]), xytext=(Vh[0][i] - x2, Vh[1][i] - y2), arrowprops=arrowprops)
    # y2+=0.05

plt.show()


