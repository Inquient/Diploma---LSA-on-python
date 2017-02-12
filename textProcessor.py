# -*- coding: utf-8 -*-

import re

class textProcessor:

    stopwordsList = []

    # Инициализация.
    def __init__(self):
        with open(u'stopwords.txt', "r", -1, "utf-8") as f:
            for line in f.readlines():
                self.stopwordsList.append(line.strip())     # Генерируем список стоп-слов

    # Удаление стоп-слов
    def remove_stopwords(self, text):
        text = [s.split(" ") for s in text]
        unstopsymboled_text = []
        for sentence in text:
            s = [word for word in sentence if word not in self.stopwordsList]  # удаляем стоп-символы
            unstopsymboled_text.append(s)
        return unstopsymboled_text

    def remove_garbage(self, text):
        new = []

    # Удаление знаков препинания
    def remove_symbols(self, text):
        unsymboled = []
        delete = re.compile(r"[^A-Za-zА-Яа-я0-9 ]", re.UNICODE)
        for sentence in text:
            s = delete.sub('', sentence)
            unsymboled.append(s)
        return unsymboled

    # Удалить слова, встречающиеся в единственном экземпляре
    def remove_unique(self, text, n):
        words = []
        for i in text:
            words.extend(i)         # Список, содержащий все слова из всех документов

        all = " ".join(words)       # Строка, содержащая все слова из всех документов

        newtext = ''
        for word in words:
            i = all.count(word)
            if i > n:               # Получаем строку со словами, встречающимися в текстах более n раз
                newtext += word
                newtext += ' '

        m = set(newtext.split(" "))
        m.remove('')                # Преобразуем список в множество, тем самым удаляя повторы

        l = [i for i in m]
        l.sort()                    # Сортируем полученный список
        return l                    # На выходе имеем список слов, встречающиеся в текстах более 1-го раза

    # Стеммер
    def stem(self, result):
        result = re.sub(r'(((?P<ignore>[ая])(в|вши|вшись))|(ив|ивши|ившись|ыв|ывши|ывшись))\b', '', result)
        result = re.sub(r'(ее|ие|ые|ое|ими|ыми|ей|ий|ый|ой|ем|им|ым|ом|его|ого|ему|ому|их|ых|ую|юю|ая|яя|ою|ею)\b', '', result)
        result = re.sub(r'(((?P<ignore>[ая])(ем|нн|вш|ющ|щ))|(ивш|ывш|ующ))\b', '', result)
        result = re.sub(r'(ся|сь)\b', '', result)
        result = re.sub(r'(((?P<ignore>[ая])(ла|на|ете|йте|ли|й|л|ем|н|ло|но|ет|ют|ны|ть|ешь|'
                        r'нно))|(ила|ыла|ена|ейте|уйте|ите|или|ыли|ей|уй|ил|ыл|им|ым|ен|ило|'
                        r'ыло|ено|ят|ует|уют|ит|ыт|ены|ить|ыть|ишь|ую|ю))\b', '', result)
        result = re.sub(r'(а|ев|ов|ие|ье|е|иями|ями|ами|еи|ии|и|ией|ей|ой|ий|й|иям|ям|ием|ем|'
                        r'ам|ом|о|у|ах|иях|ях|ы|ь|ию|ью|ю|ия|ья|я)\b', '', result)
        result = re.sub(r'(ейш|ейше)\b', '', result)
        result = re.sub(r'(ост|ость)\b', '', result)
        result = re.sub(r'и\b', '', result)
        result = re.sub(r'((?<=н)н)\b', '', result)
        result = re.sub(r'ь\b', '', result)
        return result

    # Создаём таблицу для вывода частоты встречаемости слов в каждом фрагменте текста
    def table_generator(self, keys, stemmed):
        table = []
        disp_table = []
        for w in keys:
            disp_row = [w, []]  # Строка таблицы - список из самого слова
            row = []            # и списка количества его вхождений в каждом фрагменте
            for fragment in stemmed:
                amount = fragment.count(w)  # Считаем число вхождений в текущем фрагменте
                disp_row[1].append(amount)  # Посчитав заносим их в список
                row.append(amount)
            disp_table.append(disp_row)
            table.append(row)               # Вносим строки в таблицу
        return table, disp_table



