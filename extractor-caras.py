#!/usr/bin/env python
# -*- coding: utf-8 -*-
#Usado para extraer caras a partir de directorio YA CLASIFICADO de fotogramas
import sys
import numpy as np
import math
import cv2
import os

def detectarCara(arcImagen = None):
    if arcImagen is None:
        return 0
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    imagen = cv2.cvtColor(cv2.imread(arcImagen), cv2.COLOR_BGR2GRAY)
    centroImagen = [imagen.shape[1]/2, imagen.shape[0]/2]	#w, h
    caras = detector.detectMultiScale(imagen, 1.3, 5)
    if len(caras) == 0:	
		caras = detector.detectMultiScale(imagen, 1.2, 5)	#Segundo intento, aceptando caras de menor tamaño
    if len(caras) == 0:
		caras = detector.detectMultiScale(imagen, 1.1, 5)	#Tercer intento. Causa error si se baja a 1.0
    if len(caras) == 1 and caras[0][1] < centroImagen[0]/3 or caras[0][1] > centroImagen[0]*1.5 or caras[0][0] < centroImagen[1]/3 or caras[0][0] > centroImagen[1]*1.5:
        np.append(caras,detector.detectMultiScale(imagen, 1.2, 5))  #Intento adicional si la única cara detectada esta muy decentrada
    if len(caras) == 1 and caras[0][0] < centroImagen[0]/3 or caras[0][0] > centroImagen[0]*1.5 or caras[0][1] < centroImagen[1]/3 or caras[0][1] > centroImagen[1]*1.5:
        np.append(caras,detector.detectMultiScale(imagen, 1.1, 5))  #Idém
    print caras, arcImagen
    distanciaCentroMinima = 9999
    resultado = [0,0,0,0]
    for (x,y,w,h) in caras:
        centroCara = [x+w/2,y+h/2] 
        distanciaCentroCara = math.sqrt(math.pow(math.fabs(centroImagen[0] -centroCara[0]),2) + math.pow(math.fabs(centroImagen[1] - centroCara[1]),2) )
        if distanciaCentroCara < distanciaCentroMinima:
            distanciaCentroMinima = distanciaCentroCara
            resultado = [x,y,w,h]  
    retorno = imagen[resultado[1]:resultado[1]+resultado[3], resultado[0]:resultado[0]+resultado[2]]
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.imshow('image',retorno)
    #cv2.waitKey(0)
    return retorno

if __name__ == "__main__": 
	if len(sys.argv) < 2:
		sys.exit()
	for dirname, dirnames, filenames in os.walk(sys.argv[1]):
		for subdirname in dirnames:
			subject_path = os.path.join(dirname, subdirname)
			nuevoSubdir = os.path.join(dirname + '-extractos', subdirname)
			print nuevoSubdir
			if not os.path.exists(nuevoSubdir):
				os.makedirs(nuevoSubdir)
			for filename in os.listdir(subject_path):
				try:	
					imagen = detectarCara(os.path.join(subject_path, filename))
					cv2.imwrite(os.path.join(nuevoSubdir,filename),imagen)
				except IOError, (errno, strerror):
					print "I/O error({0}): {1}".format(errno, strerror)
				except:
					print "Unexpected error:", sys.exc_info()[0]
					raise
			print " "

	
	
