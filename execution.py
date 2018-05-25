# -*- coding: utf-8 -*-
__author__ = 'Juan David Carrillo López'

import numpy as np
import mysql.connector
from xlrd import open_workbook
from mysql.connector import errorcode


if __name__ == '__main__':
    '''wb = open_workbook('resources/DATOS_COMPLETOS_0.0.1.xlsx')
    hojas = wb.sheets()
    conjunto_entrenamiento = []
    for hoja in hojas:
        if hoja.name == 'FUCK_FUZZY':
            for registro in range(1, hoja.nrows-1):
                valor_pareado = []
                for col in [1, 2, 34]:
                    valor_pareado.append(hoja.cell(registro, col).value)
                conjunto_entrenamiento.append(valor_pareado)

    n_regs = len(conjunto_entrenamiento)
    n_train = int(n_regs * 0.82)
    n_test = n_regs - n_train

    indices = np.arange(n_regs)
    np.random.seed(7)
    np.random.shuffle(indices)

    otro_entrena = np.array(conjunto_entrenamiento)[indices]
    caracteristicas = otro_entrena[:n_train, :2]
    objetivos = otro_entrena[:n_train, 2:3].ravel()
    x_prueba = otro_entrena[n_train:, :2]
    y_esperado = otro_entrena[n_train:, 2:3].ravel()
    '''
    config = {
        'user': 'root',
        'password': '',
        'host': '127.0.0.1',
        'port': '3306',
        'database': 'nlpresearch',
        'raise_on_warnings': True
    }

    try:
        conex = mysql.connector.connect(user='root', password='', host='127.0.0.1', port='3306',
                                        database='nlpresearch', raise_on_warnings=True)
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print('Something is wrong with your user name or password')
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print('Database does not exist')
        else:
            print(err)
    else:
        cursor = conex.cursor(buffered=True)

        sql = 'SELECT screen_name, tweet_text FROM tweets LIMIT 0, 30'
        cursor.execute(sql)
        for (t_text, t_name) in cursor:
            print('Name: {} tweet:{}'.format(t_name, t_text))
        cursor.close()
        conex.close()
