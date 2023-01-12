# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
m=2

l=int(input("Enter Length:"))
s1=input("Enter first sequence:")
s2=input("Enter second sequence:")
algn=np.zeros([l+1,l+1],dtype=int)
for i in range(len(s1)):
    for j in range(len(s2)):
        if(s1[i]==s2[j]):
            #flag=True
            algn[j+1][i+1]= algn[j][i]+m
        else:
            #flag=False
            mxm=max(algn[j][i]-1,algn[j][i+1]-1,algn[j+1][i]-1)
            if (mxm<0):
                mxm=0
            algn[j+1][i+1]=mxm
        #print(s1[i],"\t",s2[j],"\t",flag,algn[j+1][i+1])            
print(algn)  
r,c=l,l    
while(r):
    mxm1=max(algn[r][c],algn[r-1][c],algn[r][c-1])
    if(algn[r][c]==mxm1):
        r-=1
        c-=1
        print(mxm1,'->',end='')
    elif(algn[r-1][c]==mxm1):
        r-=1
        print(mxm1,'->',end='')
    elif(algn[r][c-1]==mxm1):
        c-=1
        print(mxm1,'->',end='')
    

    
