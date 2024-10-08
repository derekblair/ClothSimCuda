

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas.h>
#include <cuda_gl_interop.h>

#define kSide 24
#define kNumberParticles (kSide*kSide)
#define kNumberTriangles (2*(kSide-1)*(kSide-1))
#define kNumberFloats (3*kNumberParticles)
#define kNumberFloatsSquared (kNumberFloats*kNumberFloats)
#define kWidth 800
#define kHeight 800
#define ztable -kSide/2
#define root2 1.414213562373f



#define     kS  60.0f

#define     kSD  15.0f

#define 	kH  0.005

#define 	kG  -9.81f

#define 	kM  0.05f

#define 	kD  0.02f

#define 	kDW  0.1f

#define 	kSB  12.0f


struct State
{
    
    State();
    State(float* x, float* n);
    ~State();
    
    float* x;
    float* v;
    float* n;
    float* f;
};


struct Derivative
{
    Derivative();
    ~Derivative();
	float* dx;          
	float* dv;          
};



