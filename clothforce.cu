
#include "constants.h"

texture<float,2>  xtex;


Derivative* ka = NULL;
Derivative* kb = NULL;
Derivative* kc = NULL;
Derivative* kd = NULL;
State* tempstate = NULL;
float* wind = NULL;



__device__ inline void cpytex(float* targ,int j)
{
	int x = (j%kSide)*3;
	int y = j/kSide;
	targ[0] = tex2D(xtex,x,y);
	targ[1] = tex2D(xtex,x+1,y);
	targ[2] = tex2D(xtex,x+2,y);
}


__device__ inline void copytex(float* targ,int x,int y)
{
	targ[0] = tex2D(xtex,3*x,y);
	targ[1] = tex2D(xtex,3*x+1,y);
	targ[2] = tex2D(xtex,3*x+2,y);
}


__device__   inline float norm3(float* a)
{ 
	return sqrtf(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
}

__device__   inline float normalize3(float* a)
{
	float nrm = norm3(a);
	a[0]/=nrm;
	a[1]/=nrm;
	a[2]/=nrm;
	
	return nrm;
}

__device__   inline void sub3(float* a,float* b, float* c)
{
	c[0] = a[0]-b[0];
	c[1] = a[1]-b[1];
	c[2] = a[2]-b[2];
}

__device__   inline void add3(float* a,float* b, float* c)
{
	c[0] = a[0]+b[0];
	c[1] = a[1]+b[1];
	c[2] = a[2]+b[2];
}

__device__   inline void copy3(float* dst, float* src)
{
	dst[0] = src[0];
	dst[1] = src[1];
	dst[2] = src[2];
}

__device__   inline void cross3(float* a,float* b,float* c)
{
	float t[3];
	t[0] = a[1]*b[2] - a[2]*b[1];
	
	t[1] = a[2]*b[0] - a[0]*b[2];
	
	t[2] = a[0]*b[1] - a[1]*b[0];
	
	c[0]=t[0];
	c[1]=t[1];
	c[2]=t[2];
}

__device__   inline float dot3(float* a, float* b)
{
	return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}

__device__   inline void scalm3(float* a,float k)
{
	a[0]*=k;
	a[1]*=k;
	a[2]*=k;
}

__device__   inline float dist3(float* a,float* b)
{
	float s[3];
	s[0] = (a[0]-b[0]);
	s[1] = (a[1]-b[1]);
	s[2] = (a[2]-b[2]);
	return norm3(s);
}

__device__   inline void zero3(float* a)
{
	a[0]=0;
	a[1]=0;
	a[2]=0;
}

__device__ inline void addComponent(int i,int j,float eta,float stretch,float* targ,int cond)
{
	
	float d[3];
	float a[3];
	float b[3];
	float nrm;
	float scal;
	cpytex(a,i);
	cpytex(b,j);
	sub3(b,a,d);
	nrm = norm3(d);
	scal = cond?1.0f/nrm:0.0f;
	scalm3(d,scal);
	scalm3(d,stretch*(nrm-eta));
	add3(targ,d,targ);
}



__device__ inline void addNormal(int ix,int iy,int jx,int jy,int kx,int ky,float* targ,int cond)
{
	float d[3];
	float a[3];
	float b[3];
	float c[3];
	float nrm;
	
	copytex(a,ix,iy);
	copytex(b,jx,jy);
	copytex(c,kx,ky);
	
	sub3(c,a,c);
	sub3(b,a,b);
	cross3(b,c,d);
	nrm = cond?1.0f/norm3(d):0.0f;
	scalm3(d,nrm);
	add3(targ,d,targ);
}

__global__ void force(float* xc,float* vc,float* fc,float* nc,float* windc)
{
	
	
	int x = threadIdx.x+blockIdx.x*blockDim.x;
	int y = threadIdx.y+blockIdx.y*blockDim.y;
	int i = x+y*blockDim.x*gridDim.x;
    
	float fact;
	float a[3];
	float b[3];
	
	float nrml[3];
	float* targ=NULL;
	
	int c1 = x;
	int c2 = (kSide-1)-x;
	int c3 = y;
	int c4 = (kSide-1)-y;
	
	zero3(fc+3*i);
	zero3(nrml);
	
	addNormal(x,y,x+1,y,x+1,y+1,nrml,c2&&c4);
	addNormal(x,y,x+1,y+1,x,y+1,nrml,c2&&c4);
	addNormal(x,y,x,y-1,x+1,y-1,nrml,c2&&c3);
	addNormal(x,y,x+1,y-1,x+1,y,nrml,c2&&c3);
	addNormal(x,y,x,y+1,x-1,y+1,nrml,c1&&c4);
	addNormal(x,y,x-1,y+1,x-1,y,nrml,c1&&c4);
	addNormal(x,y,x-1,y,x-1,y-1,nrml,c1&&c3);
	addNormal(x,y,x-1,y-1,x,y-1,nrml,c1&&c3);
	normalize3(nrml);
	copy3(nc+3*i, nrml);
    
	fc[3*i+1] += kG*kM; 
	copy3(a,vc+3*i);
	sub3(a,windc,a);
	copy3(b, nc+3*i);
	fact = dot3(a,b);
	scalm3(b,-fact*norm3(a)*kDW);
	add3(b,fc+3*i,fc+3*i);
	
	copy3(a,vc+3*i);
	scalm3(a,-kD);
	add3(a,fc+3*i,fc+3*i);
	
	targ = fc+3*i;
	addComponent(i,i+1,1.0,kS,targ,c2);
	addComponent(i,i-1,1.0,kS,targ,c1);
	addComponent(i,i-kSide,1.0,kS,targ,c3);
	addComponent(i,i+kSide,1.0,kS,targ,c4);
	addComponent(i,i+1+kSide,root2,kSD,targ,c2&&c4);
	addComponent(i,i-kSide+1,root2,kSD,targ,c2&&c3);
	addComponent(i,i+kSide-1,root2,kSD,targ,c1&&c4);
	addComponent(i,i-kSide-1,root2,kSD,targ,c1&&c3);
	addComponent(i,i+2,2.0,kSB,targ,x<(kSide-2));
	addComponent(i,i-2,2.0,kSB,targ,x>1);
	addComponent(i,i-2*kSide,2.0,kSB,targ,y>1);
	addComponent(i,i+2*kSide,2.0,kSB,targ,y<(kSide-2));
	
}



__global__ void updatePositionAndVelocity(float* x,float* v,float *x0,float* v0,float* dx,float* dv,float dt)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    x[i] = x0[i] + dt*dx[i];
    v[i] = v0[i] + dt*dv[i];
}

__global__ void multiplyByScalar(float *vec,float k)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    vec[i]*=k;
}


__global__ void kungeRutta(float *x,float* v,float* dxa,float* dxb,float* dxc,float* dxd,float* dva,float* dvb,float* dvc,float* dvd)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    x[i]+=(kH/6.0f)*(dxa[i]+2.0f*dxb[i]+2.0f*dxc[i]+dxd[i]);
    v[i]+=(kH/6.0f)*(dva[i]+2.0f*dvb[i]+2.0f*dvc[i]+dvd[i]);
    
    if( i == kNumberFloats - 1)
    {
        x[3*(kSide*(kSide-1))] = 0;
        x[3*(kSide*(kSide-1))+1] = kSide - 1;
        x[3*(kSide*(kSide-1))+2] = 0;
        
        x[3*(kSide*kSide-1)] = kSide - 1;
        x[3*(kSide*kSide-1)+1] = kSide - 1;
        x[3*(kSide*kSide-1)+2] = 0;
        
        v[3*(kSide*(kSide-1))] = 0;
        v[3*(kSide*(kSide-1))+1] = 0;
        v[3*(kSide*(kSide-1))+2] = 0;
        
        v[3*(kSide*kSide-1)] = 0;
        v[3*(kSide*kSide-1)+1] = 0;
        v[3*(kSide*kSide-1)+2] = 0;
    }

}

void evaluate(State* initial, float dt,Derivative* d,Derivative* output)
{
	if(d!=NULL)
	{
        updatePositionAndVelocity<<<kNumberFloats/32,32>>>(tempstate->x,tempstate->v,initial->x,initial->v,d->dx,d->dv,dt);
	}
	else {
        cudaMemcpy(tempstate->x,initial->x,sizeof(float)*kNumberFloats,cudaMemcpyDeviceToDevice);
        cudaMemcpy(tempstate->v,initial->v,sizeof(float)*kNumberFloats,cudaMemcpyDeviceToDevice);
	}
	
    
     cudaMemcpy(output->dx,tempstate->v,sizeof(float)*kNumberFloats,cudaMemcpyDeviceToDevice);
	
    
    dim3 blocks(kSide/4,kSide/4);
	dim3 threads(4,4);
    
	force<<<blocks,threads>>>(tempstate->x, tempstate->v, output->dv, tempstate->n,wind);
    multiplyByScalar<<<kNumberFloats/32,32>>>(output->dv,1.0f/kM);
}


void integrate(State* state)
{
	evaluate(state, 0.0f,NULL,ka);
	evaluate(state, kH*0.5f, ka,kb);
	evaluate(state, kH*0.5f, kb,kc);
	evaluate(state, kH, kc,kd);
	kungeRutta<<<kNumberFloats/32,32>>>(state->x,state->v,ka->dx,kb->dx,kc->dx,kd->dx,ka->dv,kb->dv,kc->dv,kd->dv);
    
    dim3 blocks(kSide/4,kSide/4);
	dim3 threads(4,4);
    
	force<<<blocks,threads>>>(state->x, state->v, state->f, state->n,wind);
    
    /*
    st->x[3*(kSide*(kSide-1))] = 0;
    mstate.x[3*(kSide*(kSide-1))+1] = kSide - 1;
    mstate.x[3*(kSide*(kSide-1))+2] = 0;
    
    mstate.x[3*(kSide*kSide-1)] = kSide - 1;
    mstate.x[3*(kSide*kSide-1)+1] = kSide - 1;
    mstate.x[3*(kSide*kSide-1)+2] = 0;
    
    mstate.v[3*(kSide*(kSide-1))] = 0;
    mstate.v[3*(kSide*(kSide-1))+1] = 0;
    mstate.v[3*(kSide*(kSide-1))+2] = 0;
    
    mstate.v[3*(kSide*kSide-1)] = 0;
    mstate.v[3*(kSide*kSide-1)+1] = 0;
    mstate.v[3*(kSide*kSide-1)+2] = 0;
     
     */
}



extern void launchUpdateKernel(State* st,bool win)
{
    
    static float windloc[3] = {0,0,-5};
    
	if(win)
	{
		cudaMemcpy(wind,windloc,sizeof(float)*3,cudaMemcpyHostToDevice);
	}
    else{
		cudaMemset(wind,0,3*sizeof(float));
	}
    
    integrate(st);
    
}


extern void bindTexture(float* x)
{
   
    
    cudaMalloc((void**)&(wind),3*sizeof(float));
    cudaMemset(wind,0,3*sizeof(float));
    
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	cudaBindTexture2D(NULL,xtex,x,desc,kSide*3,kSide,sizeof(float)*kSide*3);
    
    ka = new Derivative();
    kb = new Derivative();
    kc = new Derivative();
    kd = new Derivative();
    tempstate = new State();
}

extern void unbindTexture()
{
    cudaFree(wind);
	cudaUnbindTexture(xtex);
    delete ka;
    delete kb;
    delete kc;
    delete kd;
    delete tempstate;
}



