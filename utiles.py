# -*- coding: utf-8 -*-
__author__ = 'Juan David Carrillo López'

import socket
import socketserver


def limpiarcadena(cadena):
    retorno = "\n"
    tabulacion = "\t"

    if tabulacion in cadena:
        cadena = cadena.replace(tabulacion, " ")
    elif retorno in cadena:
        cadena = cadena.replace(retorno, " ")

    return cadena


def quitaracentos(simbolo_char):
    acentos = leerarchivo("resources/acentos.txt")
    caracter_tilde = acentos[0].split(",")
    caracter_atilde = acentos[1].split(",")
    
    try:
        posicion = caracter_tilde.index(simbolo_char)
        simbolo_char = caracter_atilde[posicion]
    except IndexError as e:
        print('Indice fuera del rango {0}'.format(e))
    
    return simbolo_char


def leerarchivo(path_archivo):
    contenido = []
    
    f = open(path_archivo)
    linea = f.readline()
    
    while linea != "":
        linea = limpiarcadena(linea)
        contenido.append(linea)
        linea = f.readline()
       
    f.close()
    
    return contenido


def guardararchivo(info_arr, path_archivo, modo='w'):
    """
    :param info_arr: Matriz con la información a guardar
    :param path_archivo: Ruta y nombre del archivo
    :return:
    """
    info_arr, nombre_archivo = tuple(info_arr), path_archivo

    archivo = open(nombre_archivo, modo)
    for renglon in info_arr:
        archivo.write(renglon+'\n')
    archivo.close()


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

    
class MyTCPHandler(socketserver.BaseRequestHandler):
    """
    classdocs
    """
    
    def handle(self):
        datos = self.request.recv(4096).strip()
        #print("{0} wrote:".format(self.client_address[0]))
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
        self.servidor = socketserver.TCPServer((host, port), MyTCPHandler)
        
        # Activamos el servidor
        # mantendrá la ejecución hasta interrumpir el programa con Ctrl + C
        self.servidor.serve_forever()