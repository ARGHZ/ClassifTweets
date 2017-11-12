# -*- coding: utf-8 -*-
import csv
import socket
import SocketServer
import json

import numpy as np
from scipy.stats import itemfreq

__author__ = 'Juan David Carrillo López'


def getnewdataset():
    with open('recursos/bullyingV3/tweet.json') as json_file:
        for line in json_file:
            json_data = (json.loads(line)['id'], str(json.loads(line)['text']))
    return json_data


def votingoutputs(temp_array):
    index_outputs = []
    for col_index in range(temp_array.shape[1]):
        item_counts = itemfreq(temp_array[:, col_index])
        max_times = 0
        for class_label, n_times in item_counts:
            if n_times > max_times:
                last_class, max_times = class_label, n_times
        #  print 'feature {} class voted {} - {}'.format(col_index, class_label, n_times)
        index_outputs.append((col_index, class_label))
    return np.array(index_outputs)


def binarizearray(temp_array):
    new_array = []
    for elem in temp_array:
        if elem == 3:
            elem = 1
        else:
            elem = 0
        new_array.append(elem)
    return tuple(new_array)


def leerarchivo(path_archivo):
    contenido = []
    
    f = open(path_archivo)
    linea = f.readline()
    
    while linea != "":
        contenido.append(linea)
        linea = f.readline()
       
    f.close()
    
    return contenido


def contenido_csv(path_archivo):
    # Obtenemos los datos para probar la RNA
    with open(path_archivo, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

        data = []
        for row in spamreader:
            data.append(tuple(row))

    return tuple(data)


def guardararchivo(info_arr, path_archivo, modo='w'):
    """
    :param info_arr: Matriz con la información a guardar
    :param path_archivo: Ruta y nombre del archivo
    :return:
    """
    info_arr, nombre_archivo = tuple(info_arr), path_archivo

    archivo = open(nombre_archivo, modo)
    for renglon in info_arr:
        archivo.write(renglon + '\n')
    archivo.close()


def guardar_csv(data, path_archivo):
    """
    Escribe el contenido data en archivo CSV
    :param data:
    :param path_archivo:
    :return:
    """
    with open(path_archivo, 'wb') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for line in data:
            writer.writerow(line)


class Cliente(object):
    """
    classdocs
    """

    def __init__(self, sock=None):
        """
        Constructor
        """
        self.nombre_ip = None
        if sock is None:
            self.socket_cliente = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.socket_cliente = sock        
        self.datos = ""
        
    def conectar(self, host, puerto):
        self.socket_cliente.settimeout(10)
        self.socket_cliente.connect((host, puerto))
        self.socket_cliente.settimeout(None)
                
    def __str__(self):
        return "(Cliente) Dato recibido: {0}".format(self.datos.decode(encoding="utf_8", errors="strict"))
    
    def enviarinfo(self, msg):
        msg += "\n"
        contador = 0
        while contador < len(msg):
            enviado = self.socket_cliente.send(msg[contador:].encode(encoding="utf-8", errors="strict"))
            if enviado == 0:
                raise RuntimeError("Conexión con el socket rota")
            contador = contador + enviado
    
    def recibirinfo(self):
        return str(self.socket_cliente.recv(4096), "utf-8")
    
    @staticmethod
    def nombreip():
        return socket.gethostbyname(socket.gethostname())

    
class MyTCPHandler(SocketServer.BaseRequestHandler):
    """
    classdocs
    """
    
    def handle(self):
        datos = self.request.recv(4096).strip()
        #  print("{0} wrote:".format(self.client_address[0]))
        print(str(datos, "utf-8"))
        
        # self.request es el TCP socket conectado al cliente
        # enviarmos los datos recibidos
        self.request.sendall(datos.upper())


class Servidor(object):
    """
    classdocs
    """

    def __init__(self, host, port):
        """
        Constructor
        """
        
        # creamos el servidor
        self.servidor = SocketServer.TCPServer((host, port), MyTCPHandler)
        
        # Activamos el servidor
        # mantendrá la ejecución hasta interrumpir el programa con Ctrl + C
        self.servidor.serve_forever()