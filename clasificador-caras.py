#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
sys.path.append("../..")
from facerec.feature import Fisherfaces
from facerec.distance import EuclideanDistance, ChiSquareDistance
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
from facerec.validation import KFoldCrossValidation
import numpy as np
# try to import the PIL Image module
try:
    from PIL import Image
except ImportError:
    import Image
import cPickle
import cv2
import math
import argparse

tamanioCara=[125,150] #Todas las imagenes debes ser del mismo tamañio para entrenamiento y clasificación, por lo que se redimensionan en caso de que sea necesario
arcModelo = 'modelo.pkl'    #archivo modlo por default

def read_images(path):
    """Basado en simple_example.py
    path: directorio con imagenes ya clasificadas

    Returno: lista [X, y, clases]
            X: Lista de imagenes (arrays numpy)
            y: clases numericas usadas en facerec
            clases: nombres de las clases. Actualmente nombres de los directorios
    """
    c = 0
    X,y,clases = [], [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = Image.open(os.path.join(subject_path, filename))
                    im = im.convert("L")
                    # resize to given size (if given)
                    if (im.size[0] != tamanioCara[0] or im.size[1] != tamanioCara[1]):
                        im = im.resize(tamanioCara, Image.ANTIALIAS)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
            c = c+1
            clases.append(subdirname)
    return [X,y,clases]

def entrenarModelo(dirImagenes = None, arcModelo = arcModelo):
    if dirImagenes is None:
        print dirImagenes
        return 0
    [X,y,clases] = read_images(sys.argv[2])
    modelo = PredictableModel(feature=Fisherfaces(), classifier=NearestNeighbor(dist_metric=EuclideanDistance(), k=1)) #configuración del modelo
    modelo.compute(X, y)
    pkl = open(arcModelo, 'wb')
    cPickle.dump([modelo,clases,tamanioCara],pkl)   #se usa cPickle directamente en vez de save_model para poder insertar metadata
    pkl.close()
    validacion = KFoldCrossValidation(modelo, k=10)
    validacion.validate(X, y)
    validacion.print_results()
    
def clasificarCara(arcsCara = None, arcModelo = arcModelo):
    if arcsCara is None:
        return 0
    data = cPickle.load(open(arcModelo, 'rb'))
    modelo = data[0]
    clases = data[1]
    tamanioCara = data[2]
    #im = Image.open(arcCara)
    #im = im.convert("L")
    indicesClase = []
    for arcCara in arcsCara:
        im = detectarCara(arcCara)
        if im == -1:    #no se encontro cara
            indicesClase.append(-1)
            continue
        elif im == 0:   #archivo no existe
            print arcCara;
            return 0;
        if (im.size[0] != tamanioCara[0] or im.size[1] != tamanioCara[1]):
            im = im.resize(tamanioCara, Image.ANTIALIAS)
        X = np.asarray(im, dtype=np.uint8)
        indicesClase.append(modelo.predict(X)[0])
    print indicesClase
    res = max(indicesClase, key=indicesClase.count)
    if res == -1:
        return res
    else:
        return clases[res]

def detectarCara(arcImagen = None):
    if arcImagen is None:
        return 0
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #puede reemplazarse con otro modelos de OpenCV, adentro de data/haarcascades
    imagen = cv2.cvtColor(cv2.imread(arcImagen), cv2.COLOR_BGR2GRAY)
    centroImagen = [imagen.shape[1]/2, imagen.shape[0]/2] #w, h
    caras = detector.detectMultiScale(imagen, 1.3, 5)
    if len(caras) == 0:	
		caras = detector.detectMultiScale(imagen, 1.2, 5)	#Segundo intento, aceptando caras de menor tamaño
    if len(caras) == 0:	
		caras = detector.detectMultiScale(imagen, 1.1, 5)	#Tercer intento. Causa error si se baja a 1.0
    if len(caras) == 1 and caras[0][1] < centroImagen[0]/3 or caras[0][1] > centroImagen[0]*1.5 or caras[0][0] < centroImagen[1]/3 or caras[0][0] > centroImagen[1]*1.5:
        np.append(caras,detector.detectMultiScale(imagen, 1.2, 5))  #Intento adicional si la única cara detectada esta muy decentrada
    if len(caras) == 1 and caras[0][0] < centroImagen[0]/3 or caras[0][0] > centroImagen[0]*1.5 or caras[0][1] < centroImagen[1]/3 or caras[0][1] > centroImagen[1]*1.5:
        np.append(caras,detector.detectMultiScale(imagen, 1.1, 5))  #Idém
    distanciaCentroMinima = 9999
    caraElegida = 0
    for (x,y,w,h) in caras:
        centroCara = [x+w/2,y+h/2] 
        distanciaCentroCara = math.sqrt(math.pow(math.fabs(centroImagen[0] -centroCara[0]),2) + math.pow(math.fabs(centroImagen[1] - centroCara[1]),2) )
        if distanciaCentroCara < distanciaCentroMinima:
            distanciaCentroMinima = distanciaCentroCara
            caraElegida = [x,y,w,h]
            #print caraElegida
    if caraElegida == 0:
        return -1
    retorno = imagen[caraElegida[1]:caraElegida[1]+caraElegida[3], caraElegida[0]:caraElegida[0]+caraElegida[2]]
    #cv2.namedWindow('imagen', cv2.WINDOW_NORMAL)
    #cv2.imshow('imagen',retorno)
    #cv2.waitKey(0)
    return Image.fromstring("L", retorno.shape[2::-1], retorno.tostring()) #convertir de imagen cv2 a imagen PIL

"""
Opciones
-t [Directorio] : entrenar modelo
-c [Archivo,Archivo...,Archivo]: buscar cara en archivo y clasificar. Retorna la clase mas común si se ingresan multiples archivos

-m [Archivo] : setear archivo modelo Default: modelo.pkl

-d [Int,Int] : setear dimensiones

"""
if __name__ == "__main__":
    opciones = argparse.ArgumentParser(description='Clasificador de caras')
    modo = opciones.add_mutually_exclusive_group(required=True)
    modo.add_argument('-t', dest='entrenar', action='store', metavar='directorio', help='Directorio conteniendo imagenes de entrenamiento')
    modo.add_argument('-c', dest='clasificar', action='store', metavar='archivo', nargs='+', help='Archivo(s) de imagen para clasificación')
    opciones.add_argument('-m', dest='modelo', metavar='archivo', action='store', default='modelo.pkl', help='Archivo del modelo de clasificación. Default: modelo.pkl')
    opciones.add_argument('-d', dest='dimensiones', metavar=('X', 'Y'), action='store',  type=int, nargs=2, default=[125,50], help='Ancho y alto con las que se compararan las caras. Default: 125x150')
    args = opciones.parse_args()
    tamanioCara = args.dimensiones
    arcModelo = args.modelo
    
    if args.entrenar is not None:
        ret = entrenarModelo(args.entrenar)
        if ret == 0:
            print "Error directorio de imagenes"
    elif args.clasificar is not None:
        ret = clasificarCara(args.clasificar)
        if ret == 0:
            print "Error archivo de imagen"
        elif ret == -1:
            print "No se encontró cara"
        else:
            print ret
