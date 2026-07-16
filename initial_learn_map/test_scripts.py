from train_vect_module import train_vect_module
from inference import inference_core

import numpy as np

# xh - modulo M input
# yh - binary output
# G - input group size
#
#  bmai([0; 1],[1; 0])
#
# compilation test:
#
# clc; clear;
#xh = np.array([[0,0],[0,1],[1,0],[1,1]])
#yh = np.array([[0],[0],[0],[1]])
#model,Dy = train_vect_module(xh,yh)
#print('train executed without error')
#print('Dy final:')
#print(Dy)
# ------------------
# Inference test:

#xh = np.array([[0,0],[0,1],[1,0],[1,1]])
#yh = np.array([[0],[0],[0],[1]])
#model,__ = train_vect_module(xh,yh)
#y = model['inference'](xh)
#print('model output: ')
#print(y)
#print("Número de pesos:", len(model['W']))
#print("Primeiro peso:\n", model['W'][0])
#print("Índices I:\n", model['I'])
#print("Layers L:", model['L'])

# ------------------
# Prediction and inference test:

#xh = np.array([[0,0],[0,1],[1,0],[1,1]])
#yh = np.array([[0],[0],[0],[1]])
#model,__ = train_vect_module(xh,yh)
#y = model['inference'](xh)
#expected = yh
#print('prediction:')
#print(y)
#print('expected:')
#print(expected)
#yv = y
#ev = expected
#n = min(np.asarray(yv).size,np.asarray(ev).size)
#error = sum(np.abs(yv(np.arange(1,n+1)) - ev(np.arange(1,n+1))))
#print('total error:')
#print(error)
#-------------------------------------
# FALSE

#xh = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
#yh = np.array([[0],[0],[0],[0],[0],[0],[0],[0]])
#model,__ = train_vect_module(xh,yh)
#y = model['inference'](xh)
#expected = yh
#print('prediction:')
#print(y)
#print('expected:')
#print(expected)
#yv = y
#ev = expected
#n = min(np.asarray(yv).size,np.asarray(ev).size)
#error = sum(np.abs(yv(np.arange(1,n+1)) - ev(np.arange(1,n+1))))
#print('total error:')
#print(error)

# TRUE

#xh = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
#yh = np.array([[0],[1],[1],[0],[1],[0],[0],[1]])
#model,__ = train_vect_module(xh,yh)
#y = model['inference'](xh)
#expected = yh
#print('prediction:')
#print(y)
#print('expected:')
#print(expected)
#yv = y
#ev = expected
#n = min(np.asarray(yv).size,np.asarray(ev).size)
#error = sum(np.abs(yv(np.arange(1,n+1)) - ev(np.arange(1,n+1))))
#print('total error:')
#print(error)

# NAND

xh = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
yh = np.array([[1],[1],[1],[1],[1],[1],[1],[0]])
model,__ = train_vect_module(xh,yh)
y = model['inference'](xh)
expected = yh
print('prediction:')
print(y)
print('expected:')
print(expected)
yv = y
ev = expected
n = min(np.asarray(yv).size,np.asarray(ev).size)
error = sum(np.abs(yv(np.arange(1,n+1)) - ev(np.arange(1,n+1))))
print('total error:')
print(error)

# NOR
#xh = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
#yh = np.array([[1],[0],[0],[0],[0],[0],[0],[0]])
#model,__ = train_vect_module(xh,yh)
#y = model['inference'](np.transpose(xh))
#expected = yh
#print('prediction:')
#print(y)
#print('expected:')
#print(expected)
#yv = y
#ev = expected
#n = min(np.asarray(yv).size,np.asarray(ev).size)
#error = sum(np.abs(yv(np.arange(1,n+1)) - ev(np.arange(1,n+1))))
#print('total error:')
#print(error)

# XNOR

#xh = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
#yh = np.array([[1],[0],[0],[1],[0],[1],[1],[0]])
#model,__ = train_vect_module(xh,yh)
#y = model['inference'](np.transpose(xh))
#expected = yh
#print('prediction:')
#print(y)
#print('expected:')
#print(expected)
#yv = y
#ev = expected
#n = min(np.asarray(yv).size,np.asarray(ev).size)
#error = sum(np.abs(yv(np.arange(1,n+1)) - ev(np.arange(1,n+1))))
#print('total error:')
#print(error)

# AND

#xh = np.array([[0,0],[0,1],[1,0],[1,1]])
#yh = np.array([[0],[0],[0],[1]])
#model,Deltay = mai(xh,yh)
#y = model['inference'](xh)

# XOR
# Dataset de paridade (XOR de 3 entradas)
#xh = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
#yh = np.array([[0],[1],[1],[0],[1],[0],[0],[1]])
#model,Deltay = train_vect_module(xh,yh)
#y = model['inference'](xh)

############Measure of train###############
#expected = yh
#print('Prediction:')
#print(y)
#print('Expected:')
#print(expected)

# Calcula erro
#error = sum(np.abs(y - expected))
#print('Total error:')
#print(error)

# Acurácia

#accuracy = sum(y == expected) / len(expected) * 100
#print(np.array(['Accuracy: ',num2str(accuracy),'%']))
