


#include "cloth.h"


extern void launchUpdateKernel(State* st,bool win);
extern void bindTexture(float* x);
extern void unbindTexture();


State::State()
{
    cudaMalloc((void**)&(x),kNumberFloats*sizeof(float));
    cudaMalloc((void**)&(v),kNumberFloats*sizeof(float));
	cudaMalloc((void**)&(f),kNumberFloats*sizeof(float));
    cudaMalloc((void**)&(n),kNumberFloats*sizeof(float));
    
    
    cudaMemset(v,0,kNumberFloats*sizeof(float));
    cudaMemset(f,0,kNumberFloats*sizeof(float));
}


State::State(float* x, float* n)
{
    
    this->x = x;
    this->n = n;
    
    cudaMalloc((void**)&(v),kNumberFloats*sizeof(float));
	cudaMalloc((void**)&(f),kNumberFloats*sizeof(float));
    cudaMemset(v,0,kNumberFloats*sizeof(float));
    cudaMemset(f,0,kNumberFloats*sizeof(float));
}




State::~State()
{
    if(x)cudaFree(x);
    if(v)cudaFree(v);
    if(n)cudaFree(n);
    if(f)cudaFree(f);
}




Derivative::Derivative()
{
    cudaMalloc((void**)&(dx),kNumberFloats*sizeof(float));
    cudaMalloc((void**)&(dv),kNumberFloats*sizeof(float));
    
    cudaMemset(dx,0,kNumberFloats*sizeof(float));
    cudaMemset(dv,0,kNumberFloats*sizeof(float));
}

Derivative::~Derivative()
{
    cudaFree(dx);
    cudaFree(dv);
}


Cloth::Cloth(float* x)
{
	
	windon = false;
	
	state = new State(0,0);
    
	float* temp = new float[kNumberFloats];
	for(int i = 0 ; i < kNumberParticles; i++)
	{
		temp[3*i] = (float)(i%kSide); 
		temp[3*i+1] = (float)(i/kSide);
		temp[3*i+2] = 0;
	}
    
    cudaMemcpy(x,temp,sizeof(float)*kNumberFloats,cudaMemcpyHostToDevice);
    
	delete[] temp;
	
	bindTexture(x);
}

void Cloth::toggleWind()
{
	windon = !windon;
}

void Cloth::stepForeward(float* x)
{
    state->x = x;
    state->n = (x+kNumberFloats);
	launchUpdateKernel(state,windon);
}

Cloth::~Cloth()
{
	unbindTexture();
    state->x = 0;
    state->n = 0;
	delete state;
}















