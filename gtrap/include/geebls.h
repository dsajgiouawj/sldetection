__global__ void geebls(float *blsp, unsigned int *phase1, unsigned int *phase2, float *lc, float *tu, float *offset, unsigned int *ntrue, unsigned int n, unsigned int nf, unsigned int nb, unsigned int kma,  unsigned int kmi, unsigned int kkmi, float fmin, float df){

  /*  
      blockDim.x <= nb 

      +batch structure
      batch number = blockIdx.y

      +structure of cache 
      cache[nb]: cache[j]  for j=0,...,nb-1, 
      y[nb+kmax]: cache[j+nb]  for j=0,...,nb+nbkma-1, 
      ibi[nb+kmax]: cache[j+nbnbkma] for j=0,...,nb+nbkma-1 (nbnbkma=2*nb+kma)

  */

  unsigned int nthread = blockDim.x;
  unsigned int ithread =  threadIdx.x;
  unsigned int jf = blockIdx.x;

  float rn=float(n);
  float rnb=float(nb);
  float rnthread=float(nthread);
  float rntrue = float(ntrue[blockIdx.y]);
  unsigned int rat = int(rnb/rnthread);

  if(kmi < 1){
    kmi=1;
  }
  unsigned int nbnbkma = 2*nb+kma;

  if(kkmi < MINBIN){
    kkmi=MINBIN;
  }

  float f0=fmin+df*float(jf);
  unsigned int i = 0;

  /* thread cooperating initialization for 0 to nb*/    
  for (unsigned int m=0; m<rat; m++){
    i = m*nthread+ithread;
    if (i < nb){ 
      cache[i+nb]=0.0;
      cache[i+nbnbkma]=0.0;
      /* i.e.
	 y[i] = 0.0;
	 ibi[i] = 0; 
      */

    }
  }
  __syncthreads();

  /* thread cooperating binning for 0 to n-1 */
  float ph;
  unsigned int j;

  for (unsigned int m=0; m<int(rn/rnthread); m++){
    i = m*nthread+ithread;

    /* ----------- */
    /* Fortran-wise */
    /*    float tx = tu[i+n*blockIdx.y]; */
    /* C-wise */
    float tx = tu[i*gridDim.y+blockIdx.y]; 
    /* ----------- */

    if (i < n && tx >= INVT){
      ph     = tx*f0;
      ph     = ph-float(int(ph));
      j       = unsigned int(rnb*ph);
      /* ----------- */
      /* Fortran-wise */
      /*        atomicAdd(&cache[j+nb],lc[i+n*blockIdx.y] - offset[blockIdx.y]); */
      /* C-wise */
      atomicAdd(&cache[j+nb],lc[i*gridDim.y+blockIdx.y] - offset[blockIdx.y]); 
      /* ----------- */
      atomicAdd(&cache[j+nbnbkma],1.0); 

    }
  }
  __syncthreads();

  /* Extend the arrays  ibi[]  and  y[] beyond nb by  wrapping */
  j = nb+ithread;
  if (j < nb+kma){
    unsigned int jnb = j - nb;

    cache[j+nb] = cache[jnb+nb];
    cache[j+nbnbkma] = cache[jnb+nbnbkma];
    /* i.e.
       y[j] = y[jnb];
       ibi[j] = ibi[jnb];
    */

  }
  __syncthreads();

  /* Compute BLS statistics for this period */
  float power=0.0;
  float s;
  unsigned int k, kk, nb2, jn1, jn2, s3;
  float pow, rn1, rn3;
  float maxpower=0.0;

  for (unsigned int m=0; m<rat; m++){
    i = m*nthread+ithread;
    s     = 0.0;
    k     = 0;
    kk    = 0;
    nb2   = i+kma;
    pow = 0.0;
    if (i < nb){ 
      for (unsigned int j=i; j<nb2; j++){
	k     = k+1;
	s     = s+cache[j+nb]; 
	kk      = kk+int(cache[j+nbnbkma]); 
	/* i.e.
	   s     = s+y[j]; 
	   kk      = kk+ibi[j]; 
	*/
	if(k > kmi && kk > kkmi){
	  rn1   = float(kk);
	  pow   = s*s/(rn1*(rntrue-rn1));

	  if(pow > power){
	    power = pow;
	    jn1   = i;
	    jn2   = j;
	    rn3   = rn1;
	    s3    = s;
	  }
                       
	}
      }
    }
    cache[i] = sqrt(power);
    if(sqrt(power) > maxpower){
      maxpower=sqrt(power);
    }
  }

  __syncthreads();

  /* computing thread max */
  k = nb/2;
  while (k !=0){
    for (unsigned int m=0; m<rat; m++){
      i = m*nthread+ithread;
      if(i < k){
	cache[i] = max(cache[i],cache[i+k]);
      }
    }
    __syncthreads();
    k /= 2;         
  }
  __syncthreads();

  if (ithread == 0){     
    blsp[jf+nf*blockIdx.y] = cache[0]; 
  }  

  if (cache[0]==maxpower){
    phase1[jf+nf*blockIdx.y] = jn1;
    phase2[jf+nf*blockIdx.y] = jn2;
  }

}
