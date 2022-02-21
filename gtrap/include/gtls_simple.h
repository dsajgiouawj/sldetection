__global__ void gtls(float *tlssn, float *tlsw, float *tlst0, float *tlsl, float *tlshmax, float *lc, float *tu, int nt, int nl, int nsc, float t0min, float dt, float wmax, float dw, float deltat){ 

  /*  
      +batch structure
      batch number = blockIdx.y
    
      +structure of cache 
      cache[nsc], lc in a scoop: cache[j]  for j=0,...,nsc-1 
      cache[nsc], tu - t0 in a scoop: cache[j]  for j=nsc,...,2*nsc-1 
      cache[nthread], res_max along the width (W) direction for j=2*nsc,....,2nsc+nthread-1
      cache[1], sum of lc i.e. j=2*nsc + nthread
    
  */
  
  int nthread = blockDim.x;
  int ithread =  threadIdx.x;
  int nq = gridDim.y;
  
  float rnsc=float(nsc);
  float rnthread=float(nthread);
  int i;
  int j;
  
  /* ===================================================== */
  /* thread cooperating initialization (1) */    
  /* The default value of t should be negative (FILLVAL) */
  
  for (int m=0; m<int(rnsc/rnthread-0.00001)+1; m++){
    i = m*nthread+ithread;
    if (i < nsc){ 
      cache[i]=0.0;
      cache[i+nsc]=FILLVAL;
    }
    
  }
  
  if (ithread < nthread){ 
    cache[ithread+2*nsc]=0.0;
  }
  
  if(ithread==0){
    cache[2*nsc+nthread]=0.0;
    cache[2*nsc+nthread+1]=0.0;
  }
  
  __syncthreads();
  /* ===================================================== */
  
  
  /* thread cooperating reading a scoop */
  int k = blockIdx.x*nq+blockIdx.y;
  int nsch=int(nsc/2);
  
  for (int m=0; m<int(rnsc/rnthread-0.00001)+1; m++){
    
    i = m*nthread+ithread;
    j = k+nq*(i - nsch);
    if (i >= 0 && i < nsc && j < nq*nt && j >= 0){
      /* C-wise */
      atomicAdd(&cache[i],lc[j]);  
      /* remove FILLVAL from tu */
      atomicAdd(&cache[i+nsc],tu[j]-FILLVAL); 
    }
    
  }
  
  /* - Compute chisq */
  float W = wmax - dw*float(ithread); 
  float dl = W*0.45/float(nl);
  float L = 0.0;
  float Lmax = 0.0;
  float Hmaxmax = 0.0;
  
  float res;
  float res_max;
  float ele;
  float xAEC=0.0;
  float xBD=0.0;
  float xtB=0.0;
  float xtD=0.0;
  float t2BD=0.0;
  float tB=0.0;
  float tD=0.0;
  
  float xtmp=0.0;
  float tStart;
  float tEnd;
  float tAB;
  float tBC;
  float tCD;
  float tDE;

  /* non mask n */
  int nSx=0;
  int nAx=0;
  int nBx=0;
  int nCx=0;
  int nDx=0;
  int nEx=0;

  /* masked n */
  int nA=0;
  int nB=0;
  int nC=0;
  int nD=0;
  int nE=0;

  float tnow;
  float denA;
  float numB;
  float Hmax;
  float Hmaxh;

  res_max=0.0;

  tEnd = 0.5*W;
  tStart = - 0.5*W;

  __syncthreads();

  /* shifting time */
  for (int m=0; m<int(rnsc/rnthread-0.00001)+1; m++){

    i = m*nthread+ithread;

    if( i >= 0 && i < nsc && cache[nsc+i] >= 0.0){
      cache[i+nsc] = cache[i+nsc] - deltat*float(blockIdx.x);
    }else if(i>=0.0 && i < nsc && cache[nsc+i] < 0.0){
      /* NO NEED for FAST VERSION */
      cache[i+nsc] = -1000.0;
      cache[i] = 0.0;
    }
                
  }

  __syncthreads();

  /* get mean for each thread */
  float mean=0.0;
  float nsum=0.0;
  for (int m=0; m<nsc; m++){

    if(cache[m+nsc]>=tStart && cache[m+nsc]<=tEnd){
      mean = mean + cache[m];
      nsum = nsum + 1.0;
    }

  }

  if(nsum>0){
    mean=mean/nsum;
  }

        
  __syncthreads();


  /* =============================== */
  /* main loop for l */
  /* =============================== */

  for (int mm=0; mm<nl; mm++){
    /* =============================== */
    /*     L = float(mm+1)*dl; */
    /* =============================== */

    L = float(mm)*dl;

    xAEC=0.0;
    xBD=0.0;
    xtB=0.0;
    xtD=0.0;
    t2BD=0.0;
    tB=0.0;
    tD=0.0;

    xtmp=0.0;
    nA=0;
    nB=0;
    nC=0;
    nD=0;
    nE=0;
    nSx=0;
    nAx=0;
    nBx=0;
    nCx=0;
    nDx=0;
    nEx=0;

    /* -- boundary */
    tAB = - W*0.25 - L*0.5;
    tBC = - W*0.25 + L*0.5;
    tCD = - tBC;
    tDE = - tAB;
    
    /* ============================================== */
    /* -- search the starting -- */

    for (int i=0; i<nsc; i++){
      
      if(cache[i+nsc] > tStart){
	break;
      }
      nSx=nSx+1;

    }


    /* ============================================== */
    /* -- search the region A -- */

    for (int i=0; i<nsc-nSx; i++){

      tnow=cache[i+nSx+nsc];
      if(tnow > tAB){
	break;
      }

      if(tnow >= tStart){
	nA = nA+1;
	xAEC = xAEC + cache[i] - mean;
      }
      nAx=nAx+1;

    }


    /* ============================================== */
    /* -- search the region B -- */

    for (int i=0; i<nsc-nAx-nSx; i++){

      tnow=cache[i+nSx+nAx+nsc];

      if(tnow > tBC){
	break;
      }

      if(tnow >= tStart){
	nB = nB+1;
	xtmp=cache[i+nSx+nAx] - mean;
	xBD = xBD + xtmp;
	tB = tB + tnow;
	xtB = xtB + xtmp*tnow;
	t2BD = t2BD + tnow*tnow; 
      }
      nBx=nBx+1;

    }

    /* ============================================== */
    for (int i=0; i<nsc-nAx-nBx-nSx; i++){

      tnow=cache[i+nSx+nAx+nBx+nsc];

      if(tnow > tCD){
	break;
      }

      if(tnow >= tStart){
	nC = nC+1;
	xAEC = xAEC + cache[i+nSx+nAx+nBx] - mean;
      }
      nCx=nCx+1;



    }

    /* ============================================== */
    for (int i=0; i<nsc-nSx-nAx-nBx-nCx; i++){

      tnow=cache[i+nSx+nAx+nBx+nCx+nsc];
      if(tnow > tDE){
	break;
      }

      if(tnow >= tStart){
	nD = nD+1;
	xtmp=cache[i+nSx+nAx+nBx+nCx] - mean;
	xBD = xBD + xtmp;
	tD = tD + tnow;
	xtD = xtD + xtmp*tnow;
	t2BD = t2BD + tnow*tnow; 
      }
      nDx=nDx+1;

    }

    /* ============================================== */
    
    for (int i=0; i<nsc-nAx-nBx-nCx-nDx-nSx; i++){

      tnow=cache[i+nSx+nAx+nBx+nCx+nDx+nsc];
      if(tnow > tEnd){
	break;
      }

      if(tnow >= tStart){
	nE = nE+1;
	xAEC = xAEC + cache[i+nSx+nAx+nBx+nCx+nDx] - mean;
      }

      nEx=nEx+1;
    }

    /* filling factor of the data */
    float filfac = float(nA+nB+nC+nD+nE)/(W/deltat);

    /* ============================================== */
    if(filfac > FILFACCRIT){

      /* -- Determine H tilde */
      denA=4.0*L*L*float(nA-nC+nE)+W*W*float(nB+nD)+8.0*W*(tB-tD)+16.0*t2BD;
      numB=8.0*L*L*xAEC-4.0*W*L*xBD+16.0*L*(xtD-xtB);
      Hmax=-numB/denA;
      Hmaxh=Hmax/2.0;

      /* -- search the region A to E*/
      res=0.0;
      i=0;
      ele=0.0;

      for (int i=nSx; i<nSx+nAx; i++){
	if(cache[i+nsc] >= tStart){
	  ele=cache[i] - mean +Hmaxh;
	  res = res+ele*ele;
	}
      }

      for (int i=nSx+nAx; i<nSx+nAx+nBx; i++){
	if(cache[i+nsc] >= tStart){
	  ele=cache[i]  - mean - Hmax/L*(cache[i+nsc]+W/4.0);
	  res = res+ele*ele;
	}
      }

      for (int i=nSx+nAx+nBx; i<nSx+nAx+nBx+nCx; i++){
	if(cache[i+nsc] >= tStart){
	  ele=cache[i] - mean -Hmaxh;
	  res = res+ele*ele;
	}
      }

      for (int i=nSx+nAx+nBx+nCx; i<nSx+nAx+nBx+nCx+nDx; i++){
	if(cache[i+nsc] >= tStart){
	  ele=cache[i] - mean + Hmax/L*(cache[i+nsc]-W/4.0);
	  res = res+ele*ele;
	}
      }

      for (int i=nSx+nAx+nBx+nCx+nDx; i<nSx+nAx+nBx+nCx+nDx+nEx; i++){
	if(cache[i+nsc] >= tStart){
	  ele=cache[i] - mean +Hmaxh;
	  res = res+ele*ele;
	}
      }

      if(res>0.0){
	/* (S/N)**2 = height*height/(chisq/dof) */
	res=Hmax/sqrt(res/(nA+nB+nC+nD+nE-3)); 

	if(res > res_max){
	  res_max = res;
	  Lmax = L;
	  Hmaxmax = Hmax;

	}
      }

      /* the end of if filfac */ 
    }

    /* the end of the main loop */
  } 

  /* input maximum res_max into cache */
  cache[2*nsc+ithread]=res_max;

  __syncthreads();

  /* computing thread max */
  j = nthread/2;
  while (j !=0){
    if(ithread < j){
      cache[2*nsc+ithread] = max(cache[2*nsc+ithread],cache[2*nsc+ithread+j]);
    }
    __syncthreads();
    j /= 2;         
  }
  __syncthreads();

  /* input it to the global memory */ 
  if (ithread==0){
    tlssn[k] = cache[2*nsc];
  }

  if(res_max==cache[2*nsc]){
    tlst0[k]=deltat*float(blockIdx.x);
    tlsw[k]=W;  
    tlsl[k]=Lmax;  
    tlshmax[k]=Hmaxmax;  

  }

}
