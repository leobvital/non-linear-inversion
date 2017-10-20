# -*- coding: utf-8 -*-
"""
Created on Sun May 21 09:41:52 2017

@author: Felipe
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy

def histeq(data):
    from numpy import histogram, interp, max, size, sqrt
    nbins=int(sqrt(size(data)))   
    # Histogram
    hist,bins = histogram(data.flatten(),nbins,normed=True)
    cdf = hist.cumsum() # Cumulative distribution function
    cdf = max(data) * cdf / cdf[-1] # Normalize
    #interpolation
    return interp(data.flatten(),bins[:-1],cdf).ravel() 

def monopole2(zo,mag):
	xo,yo=0.1,0.1
	Z=0 #altura de voo
	#parametros do grid
	shape=(100,100)
	area=[0,0.2,0,0.2]
	CM = 1e-7
	T2NT = 1e9
	nx,ny=shape
	x1,x2,y1,y2=area
	xs=np.linspace(x1,x2,nx)
	ys=np.linspace(y1,y2,ny)
	Y,X=np.meshgrid(ys,xs)
	#distancia da fonte ao observador
	anom=np.zeros(Y.shape)
	x=X-xo
	y=Y-yo
	z=Z-zo
	r2 = x**2+y**2+z**2
	r3 = np.sqrt(r2)**3
	bz = z/r3    
	anom = -mag * bz
	anom *= CM * T2NT
	return anom	

##########CRIAR O DADO#############
xo_m,yo_m,zo_m,mag_m=0.1,0.1,0.1,0.02
anom2=monopole2(zo_m,mag_m)	
shape=(100,100)
area=[0,0.2,0,0.2]

nx,ny=shape
x1,x2,y1,y2=area
xs=np.linspace(x1,x2,nx)
ys=np.linspace(y1,y2,ny)
Y,X=np.meshgrid(ys,xs)

plt.figure(figsize=(8,8))
plt.contourf(Y,X, anom2, 30, cmap=mpl.cm.jet)
plt.colorbar()
plt.title('Anomalia')

##########CRIAR FUNCAO PHI#############
#para o plot	
N_p1=20

zo_plot_vet=np.linspace(-0.1, 0.3, N_p1)
m_plot_vet=np.linspace(-0.03, 0.03, N_p1)
zo_plot, m_plot = np.meshgrid(zo_plot_vet,m_plot_vet)
zo_plot_vet=zo_plot.ravel()
m_plot_vet=m_plot.ravel()
phi=np.zeros_like(zo_plot)

########### Cria a funcao phi
for i in range(N_p1):
	func=monopole2(zo_plot_vet[i],m_plot_vet[i])
	res=anom2-func
	phi[i]=np.sum(res*res)

phi=np.reshape(phi,(N_p1,N_p1))
#phi=histeq(phi)
#phi=np.reshape(phi,(N_p1,N_p1))	
fig=plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
plt.contourf(zo_plot,m_plot,phi,cmap='jet')	
#plt.plot(zo_est,m_est,'ko-')
#plt.plot(zo_est,m_est,'k-')
plt.colorbar()
#ax.text(-2,2,'p1 est=%0.5f  \np2 est=%0.5f \niteracoes=%d' 
#			%(zo_atual,m_atual,i), fontsize=15)
plt.title('Monopolo - steepest descent')
plt.xlabel('zo')
plt.ylabel('m')
	
plt.show()

