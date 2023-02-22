#!/usr/bin/env python
# coding: utf-8

# In[462]:


import numpy as np
import math
import scipy
import scipy.linalg  
import time
import scipy.sparse as sparse
import scipy.sparse.linalg


# In[463]:


Nvm = 10
nvm=10
#pre assign some stuff (Kbb), basically everything reliant on domain_K that i call a bunch
    #done
#make sure all matrix matrix vector mult does matrix vector first
    #done
#remove kdi, ldd
    #done
#get rid of several loops
    #done
#replace fd with fb, or just get rid of fd and see if it works
    #fd removed
#change all the things that I -1 to instead not need to be shifted each time
    #done
#replace zeros and empty matrices with sparse matrices
    #scipy sparse is only implemented in 2d, I could rewrite sparse form manually if needed
    #implement sparse


# In[464]:


def Fc(x):
    global lb #why only this
   # global lnv
    #global ng
   # global DomNum
    #global domain_B
    #global domain_KiiR
    #global domain_K
   # global lii
    
    y = np.zeros([ng, 1]);
    yy = np.zeros([lnv, 1]);
    w = np.zeros([lnv, 1]);
    lb =lb.astype('int')
    
    for i in range(DomNum):
        w[lb, :] = domain_B[i][lb, :]@x;
        temp = scipy.sparse.linalg.spsolve(domain_KiiR[i].T, domain_Kbi[i][:,:].T@w[lb])
        temp = scipy.sparse.linalg.spsolve(domain_KiiR[i], temp).reshape((temp.size, 1))
        yy[lb, :] = domain_Kbb[i][ :, :]@w[lb, :]-domain_Kbi[i][:,:]@temp
        y = y+domain_B[i][lb, :].T@yy[lb];
    return y


# In[465]:


def pc(x):
    y1 = np.zeros([ng, 1])
    y2 = np.zeros([nc,1])
    yc = np.zeros([nc, 1])
    y3 = np.zeros([ng,1])
    
    for i in range(DomNum):
        lx = np.zeros([lnv,1])
        llx = np.zeros([lnv,1])
        
        lx[lb] = domain_Dbb[i][:, :]@(domain_B[i][lb,:]@x) 
        ytemp=scipy.sparse.linalg.spsolve(domain_KrrR[i].T, lx[lrr])
        ytemp.reshape(ytemp.size, 1)
        ytemp = scipy.sparse.linalg.spsolve(domain_KrrR[i], ytemp)  
        llx[lrr] = ytemp.reshape(ytemp.size, 1)
        
        y1 = y1+domain_B[i][lb,:].T@(domain_Dbb[i][:,:].T@llx[lb])
        lx[lcc]=lx[lcc]-domain_Krc[i][:,:].T@ytemp.reshape(ytemp.size, 1)
        yc = yc+domain_Bc[i][lb, :].T@lx[lb]
    y2 = np.linalg.solve(Rc.T, yc)
    y2 = np.linalg.solve(Rc, y2)
    for i in range(DomNum):
        lx = np.zeros([lnv,1])
        llx = np.zeros([lnv,1])
        ytempc = domain_Bc[i][lcc,:]@y2
        lx[lcc]=ytempc
        ytemp = domain_Krc[i][:,:]@ytempc
        ytemp = scipy.sparse.linalg.spsolve(domain_KrrR[i].T, ytemp)
        ytemp = ytemp.reshape(ytemp.size, 1)
        ytemp = scipy.sparse.linalg.spsolve(domain_KrrR[i], ytemp)
        lx[lrr] = -ytemp.reshape(ytemp.size, 1)
        y3 = y3+domain_B[i][lb,:].T@(domain_Dbb[i][:, :].T@lx[lb])
    y=y1+y3
    return y
    


# In[ ]:





# In[ ]:





# In[466]:


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
        


# In[467]:


range(0)


# In[468]:


def BDDC(Nvm, nvm, a1, a2):
    #--------These constant for grid ------------     
    sqr15 = math.sqrt(15)
    intx=np.array([1/3, (6+sqr15)/21, (9-2*sqr15)/21, (6+sqr15)/21, (6-sqr15)/21, (9+2*sqr15)/21, (6-sqr15)/21]);

    inty=np.array([1/3, (6+sqr15)/21, (6+sqr15)/21, (9-2*sqr15)/21, (6-sqr15)/21, (6-sqr15)/21, (9+2*sqr15)/21]);

    intw=np.array([9/80, (155+sqr15)/2400, (155+sqr15)/2400, (155+sqr15)/2400, (155-sqr15)/2400, (155-sqr15)/2400, (155-sqr15)/2400]);

    #======================================================
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
    #global nr
    global ng
    #global nr1
    #global ldd
    global lcc
    global domain_B
    global domain_KiiR
    #global domain_K
    global nc
    global domain_D
    global Rc
    global domain_KrrR
    #global domain_K
    global domain_Bc
    global domain_Kbi
    global domain_Kbb
    global domain_Krc
    global domain_Dbb
    
    
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
    nvn01 = nvn-1
    Nvn01 = Nvn-1

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

    lijtk=np.reshape(np.linspace(1,lnv,num=lnv),(nvn1,nvm1))-1;
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
    lii =  np.reshape((lijtk[1:nvm,1:nvn]),(1,(nvn01)*(nvm01))) ;
    lii = lii #done
    lb=np.union1d(lijtk[0,:], lijtk[nvm,:])
    lb=np.union1d(lb, lijtk[1:nvm,0]);
    lb=np.union1d(lb, (lijtk[1:nvm,nvn]).T);
    #lb=lb #done
    #ldd= lb;
    lrr= np.append(lii[0,:], [lb]); #done
    lcc=np.array([lijtk[0,0], lijtk[0,nvn], lijtk[nvm,0], lijtk[nvm,nvn]]);
    lrr=np.setdiff1d(lrr,lcc);
    lrr = lrr.astype('int')
    #ldd=np.setdiff1d(ldd,lcc);
    ss=lrr.size;
    lnr=ss;
    ss=lii.size;
    lni=ss;
    ss=lb.size;
    lnb=ss;
    #nr=(nvm01)*Nvn*Nvm01*2;
    nc=(Nvm01)*Nvm01;

    ng=nvm01*Nvn*Nvm01*2+Nvm01*Nvm01;
    dsize=Nvm*Nvn
    dindex=-1;
    domain_num = np.empty(dsize)
    domain_utrue = np.empty([dsize,lnv])
    domain_u = np.empty([dsize,lnv])
    domain_rho = np.empty(dsize)
    #domain_K = np.empty([dsize, eachnv, eachnv])
    #domain_Kdi = np.empty([dsize, ldd.size, lii.size]) #switch kdi with kbi
    #domain_KiiR = np.empty([dsize, lii.size, lii.size])
    #domain_KrrR = np.empty([dsize, lrr.size, lrr.size])
    #domain_Kci  =np.empty([dsize, lcc.size, lii.size])
    #domain_Kcc = np.empty([dsize, lcc.size, lcc.size])
    #domain_Krc= np.empty([dsize, lrr.size, lcc.size])
    #domain_Kbi = np.empty([dsize, lb.size, lii.size])
    #domain_Kbb = np.empty([dsize, lb.size, lb.size])
    #domain_fd = np.empty([dsize, ldd.size])
    #domain_fc = np.empty([dsize, lcc.size])
    domain_fi  =np.empty([dsize, lii.size])
    domain_fb = np.empty([dsize, lb.size])
    domain_u =np.empty([dsize, lnv])
    domain_ff = np.empty([dsize, lnv, 1])
    #domain_Dbb = np.empty([dsize,lb.size, lb.size]) 
    #domain_Bc = np.empty([dsize, lnv, nc])
    #domain_K = []
    domain_D = []
    domain_Bc =[]
    domain_B = []
    domain_Kbi = []
    domain_Kcc = []
    domain_Krc = []
    domain_Kbb =[]
    domain_Dbb = []
    domain_KiiR = []
    domain_KrrR = []
    for i in range(dsize):
        #domain_K.append(sparse.lil_matrix((eachnv, eachnv), dtype=np.float32))
        
        domain_B.append(sparse.lil_matrix((lnv, ng), dtype=np.float32))
        #domain_B.append(np.zeros([lnv, ng]))
        domain_Bc.append(sparse.lil_matrix((lnv, nc), dtype=np.float32))
        #domain_Bc.append(np.zeros([lnv, nc]))
        domain_D.append(sparse.lil_matrix((lnv, lnv), dtype=np.float32))
        #domain_D.append(np.zeros([lnv, lnv]))
        

    
    
    for ii in range(Nvm):
        for jj in range(Nvn):
            #order the domain by the order
            #5 10
            #4 9 
            #3 8
            #2 7
            #1 6
            aK = sparse.csc_matrix((eachnv, eachnv), dtype= np.float32)
            #aK = np.zeros([eachnv, eachnv]);
            f= np.zeros([eachnv,1]);
            dindex = dindex+1;
            domain_num[dindex]=dindex+1;
            ebegin=(dindex)*eachne+1;
            eend= ebegin+eachne-1;
            rmb = (ii)*Hm; # begin x coord
            rme = rmb+hm*nvm; #end x coord
            rnb = (jj)*Hn;#begin y coord
            rne= rnb+hn*nvn; #end y coord
            x = np.zeros([2,lnv]);
            lijtk = lijtk.astype('int')
            for i in range(nvm1):
                x[1,lijtk[i,:]] = np.linspace(rnb,rne,num=nvn1);
                #removed a loop here
            for i in range(nvn1):
                x[0,lijtk[:,i]] = np.linspace(rmb,rme,num=nvm1);
                #removed a loop here
            b=np.zeros([lnv,1]);
            coeff = np.zeros([eachne])
            mydet = np.zeros([eachne])
            for k in range(eachne):
                n1 = int(lnconn[0,k]);
                n2 = int(lnconn[1,k]);
                n3 = int(lnconn[2,k]);
                x1 =x[0,n1];
                y1 = x[1,n1];
                x2 = x[0,n2];
                y2 = x[1,n2];
                x3 = x[0,n3];
                y3 = x[1,n3];
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
                aK[n1,n1] = aK[n1,n1]+adetb/2*(w1x*w1x+w1y*w1y);
                aK[n1,n2] = aK[n1,n2]+adetb/2*(w1x*w2x+w1y*w2y);
                aK[n1,n3] = aK[n1,n3]+adetb/2*(w1x*w3x+w1y*w3y);
                aK[n2,n1] = aK[n2,n1]+adetb/2*(w2x*w1x+w2y*w1y);
                aK[n2,n2] = aK[n2,n2]+adetb/2*(w2x*w2x+w2y*w2y);
                aK[n2,n3] = aK[n2,n3]+adetb/2*(w2x*w3x+w2y*w3y);
                aK[n3,n1] = aK[n3,n1]+adetb/2*(w3x*w1x+w3y*w1y);
                aK[n3,n2] = aK[n3,n2]+adetb/2*(w3x*w2x+w3y*w2y);
                aK[n3,n3] = aK[n3,n3]+adetb/2*(w3x*w3x+w3y*w3y);
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
                b[n1]= b[n1]+adetb*int1;
                b[n2]= b[n2]+adetb*int2;
                b[n3]= b[n3]+adetb*int3;
                
            domain_utrue[dindex, :]=np.zeros(lnv);
            for iit in range(lnv):
                xxx=x[0,iit];
                yyy = x[1,iit];
                domain_utrue[dindex,iit] = np.sin(xxx*pi)*np.sin(yyy*pi);
            domain_rho[dindex]=rho;
            #boundary treatment
            id_list=[];
            if ii==0:
                id_list= np.append(id_list,lijtk[0,:]);
                id_list = list(set(id_list))
            if ii == Nvm01:
                id_list = np.append(id_list, lijtk[nvm1-1,:]);
                id_list = list(set(id_list))
            if jj ==0:
                id_list = np.append(id_list, lijtk[:,0]);
                id_list = list(set(id_list))
            if jj==Nvn01:
                id_list = np.append(id_list, lijtk[:, nvn]);
                id_list = list(set(id_list))
            for l in range(len(id_list)):
                aK[int(id_list[l]),:]=0
                aK[:,int(id_list[l])]=0
            dd= len(id_list);
            #iidd=max(dd, 1);
            for ddd in range(dd):
                iidd=int(id_list[ddd])
                aK[iidd, iidd]=1;
                b[iidd]=0
            #domain_K[dindex][:, :] = aK;
            
            #temp_matrix = np.zeros([lb.size, lii.size]);
            #for m in range(lb.size):
            #    for n in range(lii.size):
            #        temp_matrix[m, n] = aK[int(ldd[m]-1), int(lii[0,n]-1)];
            
            temp_matrix = np.zeros([lii.size, lii.size]);
            lii = lii.astype('int')
            lrr=lrr.astype('int')
            lcc=lcc.astype('int')
            for m in range(lii.size):
                temp_matrix[m, :] = aK[lii[0,m], lii[0,:]].toarray();
            domain_KiiR.append(sparse.csc_matrix(scipy.linalg.cholesky(temp_matrix)));
            
            temp_matrix = np.zeros([lrr.size, lrr.size]);
            for m in range(lrr.size):
                temp_matrix[m, :] = aK[lrr[m], lrr[:]].toarray();
                #loop removed here
            domain_KrrR.append(sparse.csc_matrix(scipy.linalg.cholesky(temp_matrix)));
            
            temp_matrix = np.zeros([lcc.size, lii.size]);
            #for m in range(lcc.size):
            #    temp_matrix[m, :] = aK[lcc[m], lii[0,:]].toarray();
                #loop removed here
                    
            #domain_Kci[dindex, :,:] = temp_matrix;
            
            temp_matrix = np.zeros([lcc.size, lcc.size]);
            for m in range(lcc.size):
                temp_matrix[m, :] = aK[lcc[m], lcc[:]].toarray();
                #loop removed here
                    
            domain_Kcc.append(sparse.csc_matrix(temp_matrix));
            
            temp_matrix = np.zeros([lrr.size, lcc.size]);
            for m in range(lrr.size):
                temp_matrix[m, :] = aK[lrr[m], lcc[:]].toarray();
                #loop removed here
                    
            domain_Krc.append(sparse.lil_matrix(temp_matrix))
            
            
            temp_matrix = np.zeros([lb.size, lb.size]);
            lb= lb.astype('int')
            
            for j in range(lb.size):
                temp_matrix[j, :]= aK[lb[j], lb[:]].toarray()
                
            domain_Kbb.append(sparse.csc_matrix(temp_matrix))
            
            temp_matrix = np.zeros([lb.size, lii[0,:].size])
            for j in range(lb.size):
                temp_matrix[j, :] = aK[lb[j], lii[0,:]].toarray()
            domain_Kbi.append(sparse.csc_matrix(temp_matrix))
 
            #domain_Kbi[dindex, :, :] = temp_matrix
                
            #temp_matrix = np.zeros([ldd.size]);
            #for m in range(ldd.size):
            #    temp_matrix[m] = b[int(ldd[m]-1)];
            #domain_fd[dindex-1, :] = temp_matrix;
            
            #temp_matrix = np.zeros([lcc.size]);
            #for m in range(lcc.size):
            #    temp_matrix[m] = b[int(lcc[m]-1)];
            #domain_fc[dindex-1, :] = temp_matrix;
            
            domain_fi[dindex, :] = b[lii[0,:]][:,0];
            
            
            #loop removed here
            domain_fb[dindex, :] = b[lb][:,0];
            
            domain_u[dindex, :] = np.zeros(lnv);
            domain_ff[dindex,:] = b;
    # first get the matrix B --------------------------
    #  I order the Lagrange Multiplier as the order:


    #---   3 ---  6 ---  9
    # 11      13    15
    #---   2 ---  5 ---  8
    # 10      12    14
    #---   1 ---  4 ---  7
    #nr=0;
    
    #for i in range(DomNum):
     #   domain_K[i] = sparse.csr_matrix(domain_K[i])
    ng=-1;
    id1=lijtk[nvm1-1,1:nvn].astype('int');
    id2=lijtk[0 ,1:nvn].astype('int');
    for i in range(1, Nvm):
        for j in range(1, Nvn+1):
            domIndexL = (i-1)*Nvn+j-1;
            domIndexR = domIndexL+Nvn;
            rho1 = domain_rho[domIndexL];
            rho2 = domain_rho[domIndexR];
            rhos = rho1+rho2;
            rho1 = rho1/rhos
            rho2 = rho2/rhos
            for jj in range(0, nvn01):
                #nr = nr+1;
                ng = ng+1;
                domain_B[domIndexL][id1[jj], ng]=1;
                domain_B[domIndexR][id2[jj], ng]=1;
                domain_D[domIndexL][id1[jj], id1[jj]] = rho1;
                domain_D[domIndexR][id2[jj], id2[jj]] = rho2;
    id1 = lijtk[1:nvm, nvn].astype('int');
    id2 = lijtk[1:nvm, 0].astype('int');
    for i in range(1, Nvm+1):
        for j in range(1, Nvn):
            domIndexD = (i-1)*Nvn+j-1;
            domIndexU = domIndexD+1;
            rho1 = domain_rho[domIndexD];
            rho2 = domain_rho[domIndexU];
            rhos = rho1+rho2;            
            rho1 = rho1/rhos
            rho2 = rho2/rhos
            for ii in range(0, nvm01):
                #nr=nr+1;
                ng = ng+1;
                domain_B[domIndexD][id1[ii], ng]=1;
                domain_B[domIndexU][id2[ii], ng]=1;
                domain_D[domIndexD][id1[ii], id1[ii]] = rho1;
                domain_D[domIndexU][id2[ii], id2[ii]] = rho2;
    
    # for crosspoints
    nc=-1;
    # to get a global gijtk
    #Domain number
    # 3|4
    #1|2
    
    id1 = lijtk[nvm, nvn].astype('int');
    id2 = lijtk[0, nvn].astype('int');
    id3= lijtk[nvm, 0].astype('int');
    id4 = lijtk[0, 0].astype('int');
    num_test = 0
    for i in range(1, Nvm):
        indexi = i*nvm;
        for j in range(1, Nvn):
            indexj = j*nvn;
            domIndex1 = (i-1)*Nvn+j-1;
            domIndex2 = domIndex1+Nvn;
            domIndex3 = domIndex1 +1;
            domIndex4 = domIndex2+1;
            #global id
            #id = ijtk[indexi-1, indexj-1]
            #NCross = [Ncross id];
            #local id
            nc=nc+1
            ng=ng+1;
            domain_Bc[domIndex1][id1, nc]=1;
            domain_Bc[domIndex2][id2, nc]=1;
            domain_Bc[domIndex3][id3, nc]=1;
            domain_Bc[domIndex4][id4, nc]=1;
            num_test = num_test+4
            rho1 = domain_rho[domIndex1]
            rho2 = domain_rho[domIndex2]
            rho3 = domain_rho[domIndex3]
            rho4 = domain_rho[domIndex4]
            rhos=rho1+rho2+rho3+rho4;
            domain_D[domIndex1][id1, id1] = rho1/rhos;
            domain_D[domIndex2][id2, id2] = rho2/rhos;
            domain_D[domIndex3][id3, id3] = rho3/rhos;
            domain_D[domIndex4][id4, id4] = rho4/rhos;
            domain_B[domIndex1][id1, ng]=1;
            domain_B[domIndex2][id2, ng]=1;
            domain_B[domIndex3][id3, ng]=1;
            domain_B[domIndex4][id4, ng]=1;  
    temp_matrix = np.zeros([lb.size, lb.size])        
    for i in range(DomNum):
        for j in range(lb.size):
             temp_matrix[j,:] = domain_D[i][lb[j],lb].toarray()
        domain_Dbb.append(sparse.csc_matrix(temp_matrix))
            
    for i in range(DomNum):
        domain_B[i] = sparse.csr_matrix(domain_B[i]) #change to more better form for algebra
        domain_Bc[i] = sparse.csc_matrix(domain_Bc[i])
        
    #begin main part
    ng=ng+1
    nc=nc+1
    KccS = np.zeros([nc, nc]);
    dr = np.zeros([ng, 1]);
    ug = np.zeros([ng, 1]);
    lb = lb.astype('int')
    lii = lii.astype('int')
    for i in range(DomNum):
        kiir = domain_KiiR[i]
        temp_matrix = scipy.sparse.linalg.spsolve(kiir.T, domain_fi[i,:])
        lfi=scipy.sparse.linalg.spsolve(kiir, (temp_matrix.reshape(temp_matrix.size, 1) ));
    
        lfi.reshape(lfi.size, 1)
        #lii = np.reshape(lii,max(lii.shape) );
        #temp_matrix = np.zeros([max(lb.shape),max(lii.shape) ])
        #for lbindex in range(max(lb.shape)):
        #    domain_Kbi[i, lbindex, :] = domain_K[i][lb[lbindex], lii[0,:]].toarray()
         #   #removed loop here
        #temp_matrix = domain_Kbi[i, :, :]
        update = ((domain_B[i][lb,:]).T) @ (domain_fb[i, :]-domain_Kbi[i][:, :]@lfi)
        update_sz = update.size
        update = np.reshape(update,(update_sz,1))
        dr = dr + update
        lcc =lcc.astype('int');
        temp2 = domain_KrrR[i]
        temp_matrix = scipy.sparse.linalg.spsolve(temp2.T, domain_Krc[i]);
        temp_matrix = scipy.sparse.linalg.spsolve(domain_KrrR[i][ :, :], temp_matrix);
        KccS = KccS+domain_Bc[i][lcc, :].T@((domain_Kcc[i]-(domain_Krc[i].toarray()).T@temp_matrix)@domain_Bc[i][lcc, :]);
    Rc = scipy.linalg.cholesky(KccS);
    #input good up to this point for the below function
    [ug, error, iterate, flag]= cg(ug, dr, max_it, tol)
    #output is good now
    
    for i in range(DomNum):
        domain_u[i, lb] = (domain_B[i][lb, :]@ug)[:,0];
        #for m in range(lb.size):
             #replace with no loop to assign
        #[index2] = lb.shape
        #[index1] = lii[0, :].shape
        #temp4 = np.empty([index1, index2])
        #for j in range(index1):
        #    for k in range(index2):
        #        temp4[j, k] = domain_K[i, lii[0, j], lb[k]]
        kiir = domain_KiiR[i]
        temp4 = domain_Kbi[i][:,:].T
        ytemp = domain_fi[i,:]-temp4@domain_u[i, lb]
        #continue here
        ytemp = scipy.sparse.linalg.spsolve(kiir.T, ytemp)
        ytemp = scipy.sparse.linalg.spsolve(kiir, ytemp.reshape(ytemp.size, 1))
        domain_u[i, lii]=ytemp
        
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

    #print(4*(Nvm-1)*(Nvn-1))
    #print(A)
    #checklist = [domain_KiiR, domain_KrrR]
    #for i in range(len(checklist)):
     #   A = checklist[i]
      #  sparsity = 1.0 - ( np.count_nonzero(A) / float(A.size) )
       # print(str(i)+': '+str(sparsity))
    
                        
                
                


    
                    
            
            

            
            
            
    
                
                
                
                
                
                
                
                
                
                
                

            
            
            


# In[471]:


import tracemalloc

tracemalloc.start()

BDDC(9, 9, 9, 9)
print(tracemalloc.get_traced_memory())
tracemalloc.stop()


# In[470]:


A = np.eye(2)
A
np.linalg.solve(A, [1, 1])
A = sparse.csc_matrix(A)
scipy.sparse.linalg.spsolve(A, [1,1])


# In[ ]:




