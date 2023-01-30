#!/usr/bin/env python
# coding: utf-8

# In[50]:


import numpy as np
import math
import scipy
import scipy.linalg  


# In[ ]:





# In[51]:


Nvm = 10
nvm=10


# In[52]:


def Fc(x):
    global lb
    global lnv
    global ng
    global DomNum
    global domain_B
    global domain_KiiR
    global domain_K
    global lii
    
    y = np.zeros([ng, 1]);
    yy = np.zeros([lnv, 1]);
    w = np.zeros([lnv, 1]);
    lb =lb.astype('int')
    
    for i in range(DomNum):
        [n] = lb.shape
        [m] = lii[0, :].shape
        w[lb-1, :] = domain_B[i, lb-1, :]@x;
        Ri = domain_KiiR[i, :, :]
        temp_matrix = np.zeros([n, m])
        for j in range(n):
            for k in range(m):
                temp_matrix[j,k] = domain_K[i, lb[j]-1, lii[0,k]-1]
        temp2 = np.zeros([n,n])
        for j in range(n):
            for k in range(n):
                temp2[j,k] = domain_K[i, lb[j]-1, lb[k]-1]
        temp4 = np.zeros([m,n])
        for j in range(m):
            for k in range(n):
                temp4[j, k] = domain_K[i, lii[0, j]-1, lb[k]-1]
        temp3 = np.linalg.solve(Ri.T, temp4@w[lb-1])
        temp3 = np.linalg.solve(Ri, temp3)
        yy[lb-1, :] = temp2@w[lb-1, :]-temp_matrix@temp3
        y = y+domain_B[i, lb-1, :].T@yy[lb-1];
    return y


# In[53]:


def pc(x):
    y1 = np.zeros([ng, 1])
    y2 = np.zeros([nc,1])
    yc = np.zeros([nc, 1])
    y3 = np.zeros([ng,1])
    D_lb = np.zeros([DomNum,lb.size, lb.size])
    K_cr = np.zeros([DomNum, lcc.size, lrr.size])
    K_rc = np.zeros([DomNum, lrr.size, lcc.size])
    for i in range(DomNum):
        for j in range(lb.size):
            for k in range(lb.size):
                D_lb[i, j,k] = domain_D[i,lb[j]-1,lb[k]-1]
        for j in range(lcc.size):
            for k in range(lrr.size):
                K_cr[i,j,k] = domain_K[i,lcc[j]-1,lrr[k]-1]
        for j in range(lrr.size):
            for k in range(lcc.size):
                K_rc[i,j,k] = domain_K[i,lrr[j]-1,lcc[k]-1]
    for i in range(DomNum):
        lx = np.zeros([lnv,1])
        llx = np.zeros([lnv,1])
        
        lx[lb-1] = D_lb[i,:,:]@domain_B[i,lb-1,:]@x
        ytemp=np.linalg.solve(domain_KrrR[i,:,:].T, lx[lrr-1])
        ytemp = np.linalg.solve(domain_KrrR[i, :, :], ytemp)
        llx[lrr-1] = ytemp
        
        y1 = y1+domain_B[i,lb-1,:].T@D_lb[i,:,:].T@llx[lb-1]
        lx[lcc-1]=lx[lcc-1]-K_cr[i,:,:]@ytemp
        yc = yc+domain_Bc[i,lb-1, :].T@lx[lb-1]
    y2 = np.linalg.solve(Rc.T, yc)
    y2 = np.linalg.solve(Rc, y2)
    for i in range(DomNum):
        lx = np.zeros([lnv,1])
        llx = np.zeros([lnv,1])
        ytempc = domain_Bc[i,lcc-1,:]@y2
        lx[lcc-1]=ytempc
        ytemp = K_rc[i,:,:]@ytempc
        ytemp = np.linalg.solve(domain_KrrR[i,:,:].T, ytemp)
        ytemp = np.linalg.solve(domain_KrrR[i,:,:], ytemp)
        lx[lrr-1] = -ytemp
        y3 = y3+domain_B[i,lb-1,:].T@D_lb[i,:,:].T@lx[lb-1]
    y=y1+y3
    return y
        


# In[ ]:





# In[ ]:





# In[54]:


def cg(x, b, max_it, tol):
    flag = 0;
    iterate = 0;
    bnrm2 = np.linalg.norm(b);
    if bnrm2 == 0:
        brnm2 = 1;
    alpha = np.zeros(max_it)
    beta = np.zeros(max_it)
    d = np.zeros(max_it)
    s = np.zeros(max_it-1)
    r=b-Fc(x);
    error = np.linalg.norm(r)/bnrm2;
    if error<tol:
        return [x, error, iterate, flag]
    for iterate in range(max_it):
        z= pc(r);
        rho= r.T@z;
        if iterate>0:
            beta[iterate] = rho/rho_1;
            p = z+beta[iterate]*p;
        else:
            p=z;
        q = Fc(p);
        alpha[iterate] = rho/(p.T@q)
        x= x+alpha[iterate]*p
        r=r-alpha[iterate]*q;
        error = np.linalg.norm(r)/bnrm2;
        print('PCG residual('+ str(iterate) +') = '+str(error))
        if error<=tol:
            break
        rho_1=rho;
    d[0] = 1/alpha[0]
    for i in range(iterate-1):
        d[i+1] = beta[i+1]/alpha[i]+1/alpha[i+1]
    for i in range(iterate-1):
        s[i] = -1*((beta[i+1])**.5)/alpha[i]
    T = np.zeros([iterate, iterate])
    T[0,0] = d[0]
    for i in range(iterate-1):
        T[i+1,i+1] = d[i+1]
    for i in range(iterate-1):
        T[i,i+1] = s[i]
        T[i+1,i]  = T[i,i+1]
    lamb = np.linalg.eig(T)
    lamb = lamb[0]
    lambmax = lamb.max()
    lambmin = lamb.min()
    condnumber = lambmax/lambmin
    print('lambda max = ' + str(lambmax))
    print('lambda min = ' + str(lambmin))
    print('condition number = ' + str(condnumber))
    if error>tol:
        flag = 1
        #if 1 then no convergeance
        
    return [x, error, iterate, flag]
        


# In[55]:


range(0)


# In[56]:


def BDDC(Nvm, nvm, a1, a2):
    #--------These constant for grid ------------     
    sqr15 = math.sqrt(15)
    intx=np.array([1/3, (6+sqr15)/21, (9-2*sqr15)/21, (6+sqr15)/21, (6-sqr15)/21, (9+2*sqr15)/21, (6-sqr15)/21]);

    inty=np.array([1/3, (6+sqr15)/21, (6+sqr15)/21, (9-2*sqr15)/21, (6-sqr15)/21, (6-sqr15)/21, (9+2*sqr15)/21]);

    intw=np.array([9/80, (155+sqr15)/2400, (155+sqr15)/2400, (155+sqr15)/2400, (155-sqr15)/2400, (155-sqr15)/2400, (155-sqr15)/2400]);

    #======================================================
    import time
    start = time.time()
    #--------  This constant for PCG -------------

    max_it=1000; 
    tol=10**(-8);
    #======================================================

    rm=1;
    rn=1;

    global DomNum
    global lii
    global lb
    global lrr
    global lnv
    global nv
    global nr
    global ng
    global nr1
    global ldd
    global lcc
    global domain_B
    global domain_KiiR
    global domain_K
    global nc
    global domain_D
    global Rc
    global domain_KrrR
    global domain_K
    global domain_Bc
    
    
    
    Nvn=Nvm;
    Ne=Nvm*Nvn;
    Hm=rm/Nvm;
    Hn=rn/Nvn;
    DomNum=Nvm*Nvn;
         # order all the domain 



    #   get the mesh in each subdomain     

    nvn=nvm;
    hm=rm/(nvm*Nvm);
    hn=rn/(nvn*Nvn);
    ne=Nvm*Nvn*nvm*nvn*2;
    pi = math.pi
    nvm1=nvm+1;
    nvn1=nvn+1;
    nvm01=nvm-1;
    Nvm01=Nvm-1;

    lnv=nvm1*nvn1;
    
    # domain struct:
    #      int num;           /* global number of the domain */
    #      Mat K;             /* local stiffness matrix */
    #      Vec f;     
    #     Mat BrT;              /* transpose if the connectivity matrices;
    #     Mat Bc;               /* the corner connectivity matrix */
    #      Mat Q;               /* matrix for optional (e.g. edge) constraints */
    #      set InteriorN       /* Interior nodes */
    #      set Boundary[4]    /*  boundary nodes */

    # initial the domain structure 

    eachne=nvm*nvn*2;
    eachnv=nvn1*nvm1;
    lnv=eachnv;
    nv=lnv*Nvm*Nvn;

    #  to get the local connectivity matrix

    lijtk=np.reshape(np.linspace(1,lnv,num=lnv),(nvn1,nvm1));
    lnconn= np.zeros([3, nvm*nvn*2])
    ii=0;
    for i in range(nvm):
        ip=i+1;
        for  j in range(nvn):
            jp=j+1;
            lnconn[0,ii]  =lijtk[i,j];
            lnconn[1,ii]  =lijtk[ip,j];
            lnconn[2,ii]  =lijtk[ip,jp];
            lnconn[0,ii+1]=lijtk[i,j];
            lnconn[1,ii+1]=lijtk[i,jp];
            lnconn[2,ii+1]=lijtk[ip,jp];
            ii=ii+2;  
    lii =  np.reshape((lijtk[1:nvm,1:nvn]),(1,(nvn-1)*(nvm-1))) ;
    lb=np.union1d(lijtk[0,:], lijtk[nvm1-1,:])
    lb=np.union1d(lb, lijtk[1:nvm,0]);
    lb=np.union1d(lb, (lijtk[1:nvm,nvn1-1]).T);
    ldd= lb;
    lrr= np.append(lii[0,:], [lb]);
    lcc=np.array([lijtk[0,0], lijtk[0,nvn1-1], lijtk[nvm1-1,0], lijtk[nvm1-1,nvn1-1]]);
    lrr=np.setdiff1d(lrr,lcc);
    lrr = lrr.astype('int')
    ldd=np.setdiff1d(ldd,lcc);
    ss=lrr.size;
    lnr=ss;
    ss=lii.size;
    lni=ss;
    ss=lb.size;
    lnb=ss;
    nr=(nvm-1)*Nvn*Nvm01*2;
    nc=(Nvm-1)*Nvm01;

    ng=nvm01*Nvn*Nvm01*2+Nvm01*Nvm01;
    dsize=Nvm*Nvn
    dindex=0;
    domain_num = np.empty(dsize)
    domain_utrue = np.empty([dsize,lnv])
    domain_u = np.empty([dsize,lnv])
    domain_rho = np.empty(dsize)
    domain_K = np.empty([dsize, eachnv, eachnv])
    domain_Kdi = np.empty([dsize, ldd.size, lii.size])
    domain_KiiR = np.empty([dsize, lii.size, lii.size])
    domain_KrrR = np.empty([dsize, lrr.size, lrr.size])
    domain_Kci  =np.empty([dsize, lcc.size, lii.size])
    domain_Kcc = np.empty([dsize, lcc.size, lcc.size])
    domain_Krc= np.empty([dsize, lrr.size, lcc.size])
    domain_fd = np.empty([dsize, ldd.size])
    domain_fc = np.empty([dsize, lcc.size])
    domain_fi  =np.empty([dsize, lii.size])
    domain_fb = np.empty([dsize, lb.size])
    domain_u =np.empty([dsize, lnv])
    domain_ff = np.empty([dsize, lnv, 1])
    domain_Bc = np.empty([dsize, lnv, nc])
    domain_D = np.empty([dsize, lnv, lnv])
    domain_B = np.empty([dsize, lnv, ng])
    
    for ii in range(Nvm):
        for jj in range(Nvn):
            #order the domain by the order
            #5 10
            #4 9 
            #3 8
            #2 7
            #1 6
            aK = np.zeros([eachnv, eachnv]);
            f= np.zeros([eachnv,1]);
            dindex = dindex+1;
            domain_num[dindex-1]=dindex;
            ebegin=(dindex-1)*eachne+1;
            eend= ebegin+eachne-1;
            rmb = (ii)*Hm; # begin x coord
            rme = rmb+hm*nvm; #end x coord
            rnb = (jj)*Hn;#begin y coord
            rne= rnb+hn*nvn; #end y coord
            x = np.zeros([2,lnv]);
            for i in range(nvm1):
                for j in range(lijtk[i,:].shape[0]):
                    x[1,int(lijtk[i,j]-1)] = np.linspace(rnb,rne,num=nvn1)[j];
            for i in range(nvn1):
                for j in range(lijtk[:,i].shape[0]):
                    x[0,int(lijtk[j,i]-1)] = np.linspace(rmb,rme,num=nvm1)[j];
            b=np.zeros([lnv,1]);
            coeff = np.zeros([eachne])
            mydet = np.zeros([eachne])
            for k in range(eachne):
                n1 = int(lnconn[0,k]);
                n2 = int(lnconn[1,k]);
                n3 = int(lnconn[2,k]);
                x1 =x[0,n1-1];
                y1 = x[1,n1-1];
                x2 = x[0,n2-1];
                y2 = x[1,n2-1];
                x3 = x[0,n3-1];
                y3 = x[1,n3-1];
                if x3<=0.5:
                    if y3<=0.5:
                        rho=a1;
                    else:
                        rho=a2;
                else:
                    if y3<=0.5:
                        rho=a2
                    else:
                        rho=a1;
                coeff[k]=rho;
                b11 = x2-x1;
                b12 = x3-x1;
                b21 = y2-y1;
                b22=y3-y1;
                detb = b11*b22-b12*b21;
                adetb = abs(detb)*rho;
                mydet[k] = adetb
                d11=b22/detb;
                d12 = -b21/detb;
                d21 = -b12/detb;
                d22=b11/detb;
                w1x = -(d11+d12);
                w1y = -(d21+d22);
                w2x = d11;
                w2y = d21;
                w3x = d12;
                w3y = d22;
                aK[n1-1,n1-1] = aK[n1-1,n1-1]+adetb/2*(w1x*w1x+w1y*w1y);
                aK[n1-1,n2-1] = aK[n1-1,n2-1]+adetb/2*(w1x*w2x+w1y*w2y);
                aK[n1-1,n3-1] = aK[n1-1,n3-1]+adetb/2*(w1x*w3x+w1y*w3y);
                aK[n2-1,n1-1] = aK[n2-1,n1-1]+adetb/2*(w2x*w1x+w2y*w1y);
                aK[n2-1,n2-1] = aK[n2-1,n2-1]+adetb/2*(w2x*w2x+w2y*w2y);
                aK[n2-1,n3-1] = aK[n2-1,n3-1]+adetb/2*(w2x*w3x+w2y*w3y);
                aK[n3-1,n1-1] = aK[n3-1,n1-1]+adetb/2*(w3x*w1x+w3y*w1y);
                aK[n3-1,n2-1] = aK[n3-1,n2-1]+adetb/2*(w3x*w2x+w3y*w2y);
                aK[n3-1,n3-1] = aK[n3-1,n3-1]+adetb/2*(w3x*w3x+w3y*w3y);
                ointx = x1+b11*intx+b12*inty;
                ointy = y1+b21*intx+b22*inty;
                int1 = 0;
                int2 = 0;
                int3=0;
                for i in range(7):
                    xxx=ointx[i];
                    yyy = ointy[i];
                    ff = 2*np.sin(xxx*pi)*pi*pi*np.sin(yyy*pi);
                    int1 =int1+ff*(1-intx[i]-inty[i])*intw[i];
                    int2 = int2+ff*(intx[i])*intw[i];
                    int3 = int3+ff*(inty[i])*intw[i];
                b[n1-1]= b[n1-1]+adetb*int1;
                b[n2-1]= b[n2-1]+adetb*int2;
                b[n3-1]= b[n3-1]+adetb*int3;
                
            domain_utrue[dindex-1, :]=np.zeros(lnv);
            for iit in range(lnv):
                xxx=x[0,iit];
                yyy = x[1,iit];
                domain_utrue[dindex-1,iit] = np.sin(xxx*pi)*np.sin(yyy*pi);
            domain_rho[dindex-1]=rho;
            #boundary treatment
            id_list=[];
            if ii==0:
                id_list= np.append(id_list,lijtk[0,:]);
                id_list = list(set(id_list))
            if ii == Nvm-1:
                id_list = np.append(id_list, lijtk[nvm1-1,:]);
                id_list = list(set(id_list))
            if jj ==0:
                id_list = np.append(id_list, lijtk[:,0]);
                id_list = list(set(id_list))
            if jj==Nvn-1:
                id_list = np.append(id_list, lijtk[:, nvn1-1]);
                id_list = list(set(id_list))
            for l in range(len(id_list)):
                aK[int(id_list[l])-1,:]=0
                aK[:,int(id_list[l])-1]=0
            dd= len(id_list);
            #iidd=max(dd, 1);
            for ddd in range(dd):
                iidd=int(id_list[ddd])
                aK[iidd-1, iidd-1]=1;
                b[iidd-1]=0
            domain_K[dindex-1,:, :] = aK;
            
            temp_matrix = np.zeros([ldd.size, lii.size]);
            for m in range(ldd.size):
                for n in range(lii.size):
                    temp_matrix[m, n] = aK[int(ldd[m]-1), int(lii[0,n]-1)];
            domain_Kdi[dindex-1, :, :] = temp_matrix;
            
            temp_matrix = np.zeros([lii.size, lii.size]);
            for m in range(lii.size):
                for n in range(lii.size):
                    temp_matrix[m, n] = aK[int(lii[0,m]-1), int(lii[0,n]-1)];
            domain_KiiR[dindex-1,:,:] = scipy.linalg.cholesky(temp_matrix);
            
            temp_matrix = np.zeros([lrr.size, lrr.size]);
            for m in range(lrr.size):
                for n in range(lrr.size):
                    temp_matrix[m, n] = aK[int(lrr[m]-1), int(lrr[n]-1)];
            domain_KrrR[dindex-1, :, :] = scipy.linalg.cholesky(temp_matrix);
            
            temp_matrix = np.zeros([lcc.size, lii.size]);
            for m in range(lcc.size):
                for n in range(lii.size):
                    temp_matrix[m, n] = aK[int(lcc[m]-1), int(lii[0,n]-1)];
            domain_Kci[dindex-1, :,:] = temp_matrix;
            
            temp_matrix = np.zeros([lcc.size, lcc.size]);
            for m in range(lcc.size):
                for n in range(lcc.size):
                    temp_matrix[m, n] = aK[int(lcc[m]-1), int(lcc[n]-1)];
            domain_Kcc[dindex-1, :,:] = temp_matrix;
            
            temp_matrix = np.zeros([lrr.size, lcc.size]);
            for m in range(lrr.size):
                for n in range(lcc.size):
                    temp_matrix[m, n] = aK[int(lrr[m]-1), int(lcc[n]-1)];
            domain_Krc[dindex-1, :,:] = temp_matrix;
            
            temp_matrix = np.zeros([ldd.size]);
            for m in range(ldd.size):
                temp_matrix[m] = b[int(ldd[m]-1)];
            domain_fd[dindex-1, :] = temp_matrix;
            
            temp_matrix = np.zeros([lcc.size]);
            for m in range(lcc.size):
                temp_matrix[m] = b[int(lcc[m]-1)];
            domain_fc[dindex-1, :] = temp_matrix;
            
            temp_matrix = np.zeros([lii.size]);
            for m in range(lii.size):
                temp_matrix[m] = b[int(lii[0,m]-1)];
            
            domain_fi[dindex-1, :] = temp_matrix;
            
            temp_matrix = np.zeros([lb.size]);
            for m in range(lb.size):
                temp_matrix[m] = b[int(lb[m]-1)];
            domain_fb[dindex-1, :] = temp_matrix;
            
            domain_u[dindex-1, :] = np.zeros(lnv);
            domain_ff[dindex-1,:] = b;
            
            domain_Bc[dindex-1, :, :] = np.zeros([lnv,nc]);
            domain_D[dindex-1, :, :] = np.zeros([lnv, lnv]);
            domain_B[dindex-1, :, :] = np.zeros([lnv, ng]);
    # first get the matrix B --------------------------
    #  I order the Lagrange Multiplier as the order:


    #---   3 ---  6 ---  9
    # 11      13    15
    #---   2 ---  5 ---  8
    # 10      12    14
    #---   1 ---  4 ---  7
    nr=0;
    ng=0;
    id1=lijtk[nvm1-1,1:nvn].astype('int');
    id2=lijtk[0 ,1:nvn].astype('int');
    for i in range(1, Nvm):
        for j in range(1, Nvn+1):
            domIndexL = (i-1)*Nvn+j-1;
            domIndexR = domIndexL+Nvn;
            for jj in range(1, nvn):
                nr = nr+1;
                ng = ng+1;
                domain_B[domIndexL, int(id1[jj-1]-1), ng-1]=1;
                domain_B[domIndexR, id2[jj-1]-1, ng-1]=1;
                rho1 = domain_rho[domIndexL];
                rho2 = domain_rho[domIndexR];
                rhos = rho1+rho2;
                domain_D[domIndexL, id1[jj-1]-1, id1[jj-1]-1] = rho1/rhos;
                domain_D[domIndexR, id2[jj-1]-1, id2[jj-1]-1] = rho2/rhos;
    
    id1 = lijtk[1:nvm, nvn].astype('int');
    id2 = lijtk[1:nvm, 0].astype('int');
    for i in range(1, Nvm+1):
        for j in range(1, Nvn):
            domIndexD = (i-1)*Nvn+j-1;
            domIndexU = domIndexD+1;
            for ii in range(1, nvm):
                nr=nr+1;
                ng = ng+1;
                domain_B[domIndexD, id1[ii-1]-1, ng-1]=1;
                domain_B[domIndexU, id2[ii-1]-1, ng-1]=1;
                rho1 = domain_rho[domIndexD];
                rho2 = domain_rho[domIndexU];
                rhos = rho1+rho2;
                domain_D[domIndexD, id1[ii-1]-1, id1[ii-1]-1] = rho1/rhos;
                domain_D[domIndexU, id2[ii-1]-1, id2[ii-1]-1] = rho2/rhos;
    
    # for crosspoints
    nc=0;
    # to get a global gijtk
    #Domain number
    # 3|4
    #1|2
    
    id1 = lijtk[nvm1-1, nvn1-1].astype('int');
    id2 = lijtk[0, nvn1-1].astype('int');
    id3= lijtk[nvm1-1, 0].astype('int');
    id4 = lijtk[0, 0].astype('int');
    for i in range(1, Nvm):
        indexi = i*nvm1-1;
        for j in range(1, Nvn):
            indexj = j*nvn1-1;
            domIndex1 = (i-1)*Nvn+j-1;
            domIndex2 = domIndex1+Nvn;
            domIndex3 = domIndex1 +1;
            domIndex4 = domIndex2+1;
            #global id
            #id = ijtk[indexi-1, indexj-1]
            #NCross = [Ncross id];
            #local id
            nc=nc+1
            id=nc;
            ng=ng+1;
            domain_Bc[domIndex1, id1-1, id-1]=1;
            domain_Bc[domIndex2, id2-1, id-1]=1;
            domain_Bc[domIndex3, id3-1, id-1]=1;
            domain_Bc[domIndex4, id4-1, id-1]=1;
            rho1 = domain_rho[domIndex1]
            rho2 = domain_rho[domIndex2]
            rho3 = domain_rho[domIndex3]
            rho4 = domain_rho[domIndex4]
            rhos=rho1+rho2+rho3+rho4;
            domain_D[domIndex1, id1-1, id1-1] = rho1/rhos;
            domain_D[domIndex2, id2-1, id2-1] = rho2/rhos;
            domain_D[domIndex3, id3-1, id3-1] = rho3/rhos;
            domain_D[domIndex4, id4-1, id4-1] = rho4/rhos;
            domain_B[domIndex1, id1-1, ng-1]=1;
            domain_B[domIndex2, id2-1, ng-1]=1;
            domain_B[domIndex3, id3-1, ng-1]=1;
            domain_B[domIndex4, id4-1, ng-1]=1;     
    nr1=nr+1;
    
    #begin main part
    
    KccS = np.zeros([nc, nc]);
    dr = np.zeros([ng, 1]);
    ug = np.zeros([ng, 1]);
    lb = lb.astype('int')
    lii = lii.astype('int')
    for i in range(DomNum):
        lfi=np.linalg.solve(domain_KiiR[i, :, :], (np.linalg.solve(domain_KiiR[i,:,:].T, domain_fi[i,:]) ));
        #lii = np.reshape(lii,max(lii.shape) );
        temp_matrix = np.zeros([max(lb.shape),max(lii.shape) ])
        for lbindex in range(max(lb.shape)):
            for liiindex in range(max(lii.shape)):
                temp_matrix[lbindex, liiindex] = domain_K[i, lb[lbindex]-1, lii[0,liiindex]-1]
        
        update = (domain_B[i,lb-1,:].T) @ (domain_fb[i, :]-temp_matrix@lfi)
        update_sz = update.size
        update = np.reshape(update,(update_sz,1))
        dr = dr + update
        lcc =lcc.astype('int');
        temp_matrix = np.linalg.solve(domain_KrrR[i,:,:].T, domain_Krc[i,:,:]);
        
        temp_matrix = np.linalg.solve(domain_KrrR[i, :, :], temp_matrix);
        KccS = KccS+domain_Bc[i, lcc-1, :].T@(domain_Kcc[i]-domain_Krc[i, :, :].T@temp_matrix)@domain_Bc[i, lcc-1, :];
        
    Rc = scipy.linalg.cholesky(KccS);
    #input good up to this point for the below function
    [ug, error, iterate, flag]= cg(ug, dr, max_it, tol)
    #output is good now
    
    for i in range(DomNum):
        for m in range(lb.size):
            domain_u[i, lb[m]-1] = domain_B[i, lb[m]-1, :]@ug;
        [index2] = lb.shape
        [index1] = lii[0, :].shape
        temp4 = np.empty([index1, index2])
        for j in range(index1):
            for k in range(index2):
                temp4[j, k] = domain_K[i, lii[0, j]-1, lb[k]-1]
        ytemp = domain_fi[i,:]-temp4@domain_u[i, lb-1]
        #continue here
        ytemp = np.linalg.solve(domain_KiiR[i, :, :].T, ytemp)
        domain_u[i, lii-1]=np.linalg.solve(domain_KiiR[i, :, :], ytemp)
    resl2 = 0
    errmax = 0
    for i in range(DomNum):
        udiff = domain_utrue[i, :]-domain_u[i,:]
        resl2 = resl2+np.linalg.norm(udiff)**2
        errmax = max(max(abs(udiff)), errmax)
    resl2 = (resl2**.5)/(nvm*Nvm)
    stop = time.time()
    print('Time Elasped:' + str(stop - start))
    print('L2 = ' + str(resl2))
    print('max = '+ str(errmax ))
    
                        
                
                


    
                    
            
            

            
            
            
    
                
                
                
                
                
                
                
                
                
                
                

            
            
            


# In[57]:


BDDC(3, 4, 2, 7)


# In[58]:


A = np.array([[2,2]])
A=np.reshape(A, 2)
print(A)


# In[59]:


max(A.shape)


# In[60]:


A.astype('int')


# In[ ]:





# In[61]:


print(Nvm)


# In[ ]:





# In[ ]:





# In[ ]:




