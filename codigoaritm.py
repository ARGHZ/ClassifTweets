# -*- coding: utf-8 -*-
__author__ = 'Juan David Carrillo López'

from math import log


class CodigoAritm(object):
    """
    classdocs
    """

    def __init__(self, simbs, prob_simb):
        """
        Constructor
        """
        simbs = simbs.split(" | ")
        prob_simb = prob_simb.split(" | ")
        if len(simbs) != len(prob_simb):
            raise SimbProbsError(len(simbs), len(prob_simb))
        try:
            self.simbolos = dict(zip([ord(s) for s in simbs], prob_simb))
        except:
            raise ItemVacioError

        self.termina = simbs[len(simbs) - 1]
        inferior, superior = 0, 0
        for k, v in self.simbolos.items():
            superior = inferior + float(v)
            nvo_conj = (float(v), inferior, superior)
            self.simbolos[k] = nvo_conj
            inferior = superior
        self.tabla = {'codificar': '', 'decodificar': ''}
        self.valor = 0
        self.mensaje_base2 = 0

    def __repr__(self):
        return "{0} \n{1}\n{2}".format([chr(clave) for clave, valor in self.simbolos.items()], self.tabla['codificar'],
                                       self.tabla['decodificar'])

    def precodmsj(self, msg):
        if not (self.termina in msg):
            raise NoTerminaError(self.termina)
        inferior, superior = 0, 1

        for s in msg:
            letra_num = ord(s)
            try:
                fila = self.simbolos[letra_num]
            except KeyError as e:
                raise ExistSimbError(chr(e.args[0]))
            prob, x, y = fila

            x, y = round(x, 15), round(y, 15)
            rango = round((superior - inferior), 15)
            pivote = CodigoAritm.liminf(inferior, rango, x)
            superior = round((CodigoAritm.limsup(inferior, rango, y)), 15)
            inferior = round(pivote, 15)
            # print("{0} -|- {1} -|- {2} -|- {3} \n".format(chr(letra_num),rango,(x,y),(inferior,superior)))
            self.tabla['codificar'] += chr(letra_num) + ' -|- ' + str(rango) + ' -|- ' + str((x, y)) + ' -|- ' + str(
                (inferior, superior)) + '\n'

        self.mensaje_base2 = (CodigoAritm.puntodecimalbase2(inferior), CodigoAritm.puntodecimalbase2(superior))

    def ratioabsoluta(self):
        return log(len(self.simbolos), 2)

    def entropiadelmensaje(self, msg):
        sumatoria = 0
        for s in msg:
            letra_num = ord(s)
            try:
                fila = self.simbolos[letra_num]
            except KeyError as e:
                raise ExistSimbError(chr(e.args[0]))
            prob, x, y = fila
            sumatoria += prob * log((1/prob), 2)
        return sumatoria

    def codificarmensaje(self):
        inf_bits, sup_bits = self.mensaje_base2[0].split(".")[1], self.mensaje_base2[1].split(".")[1]
        mensaje_bits = inf_bits

        for i in range(len(inf_bits)):

            if inf_bits[i] != sup_bits[i]:
                if inf_bits[i] == 1:
                    mensaje_bits = inf_bits[:i + 1]
                else:
                    mensaje_bits = sup_bits[:i + 1]
                break

        n_bin = '0.' + mensaje_bits
        self.valor = round(CodigoAritm.basediez(n_bin), 15)
        return n_bin, self.valor

    def decodificarmensaje(self, valor_decimal):
        """Establecemos algunos valores por default
        """
        msg = ""
        inferior, superior = 0, 1
        letra_num, intervalo = 0, (0, 0)

        while True:
            cociente = round((valor_decimal - inferior) / (superior - inferior), 15)

            """Buscando el cociente entre los rangos previamente calculados
            """
            for s in self.simbolos:
                prob, x, y = self.simbolos[s]
                if x <= cociente < y:
                    letra_num, intervalo = s, (x, y)

            x, y = round(intervalo[0], 15), round(intervalo[1], 15)
            rango = round((superior - inferior), 15)

            """ Criterio para determinar cuándo terminar los cálculos
            """
            msg += chr(letra_num)
            self.tabla['decodificar'] += str((inferior, superior)) + ' -|- ' + str(rango) + ' -|- ' + str(
                cociente) + ' -|- ' + str((chr(letra_num), (x, y))) + '\n'
            if chr(letra_num) == self.termina:
                break

            pivote = CodigoAritm.liminf(inferior, rango, x)
            superior = round((CodigoAritm.limsup(inferior, rango, y)), 15)
            inferior = round(pivote, 15)

        return msg

    def gettablacodificar(self):
        return self.tabla['codificar']

    def gettabladecodificar(self):
        return self.tabla['decodificar']

    def simbtermina(self):
        return self.termina

    @staticmethod
    def liminf(inf, rango, x):
        return inf + rango * x

    @staticmethod
    def limsup(inf, rango, y):
        return inf + rango * y

    @staticmethod
    def puntodecimalbase2(flotante):
        dec_binario = "0."
        iteracion = 0
        unidad = str(flotante).split('.')[0]

        # Verificando que no exista una parte entera mayor a 0
        if int(unidad) == 0:
            try:
                while flotante != 0:
                    iteracion += 1
                    num_flot = flotante * 2
                    entero, decimales = str(num_flot).split(".")
                    dec_binario += str(entero)

                    flotante = float("0." + decimales)

                    if iteracion == 128:
                        break
            except:
                dec_binario += '0'
        else:
            dec_binario += '0'

        return dec_binario

    @staticmethod
    def basediez(base_2):
        base_2 = base_2.split('.')
        result, potencia = 0, 0

        for i in base_2[1]:
            potencia -= 1
            result += int(i) * (2 ** potencia)

        return result


class SimbProbsError(Exception):
    def __init__(self, n_simbs, n_probs):
        self.n_simbs = n_simbs
        self.n_probs = n_probs

    def __str__(self):
        return 'Hay {0} simbolos para {1} probabilidades correspondientes'.format(self.n_simbs, self.n_probs)


class NoTerminaError(Exception):
    def __init__(self, termina):
        self.caract_termina = termina

    def __str__(self):
        return 'No se encontró el caracter de terminación: {0}'.format(self.caract_termina)


class ExistSimbError(Exception):
    def __init__(self, desconocido):
        self.desc_simb = desconocido

    def __str__(self):
        return 'El símbolo: {0} no se encuentra en el alfabeto'.format(self.desc_simb)


class ItemVacioError(Exception):
    def __init__(self):
        pass

    def __str__(self):
        return 'Se encontró un elemento vacío al crear el arreglo'