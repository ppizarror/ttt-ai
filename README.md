# Proyecto

El archivo ttt_ai.cpp es el codigo principal, en el se entrena la red mediante un algoritmo genetico,
el siguiente comando permite compilarlo:

* g++ ttt_ai.cpp Tinn.c -lOpenCL -std=c++11

El archivo Tinn.c es la implementacion de una red neuronal simple en C, se compila con lo siguiente:

* g++ test.cpp Tinn.c -lOpenCL -std=c++11

El archivo test.cpp permite jugar con la red neural generada por ttt_ai.cpp

mejor_red es un archivo que contiene una red ya generada por ttt_ai.cpp y esta lista para ser usada por test.cpp
