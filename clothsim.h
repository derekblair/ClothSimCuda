

#include <GL/glew.h>
#include "cloth.h"


class ClothSim
{
public:
	ClothSim(int argc, char **argv);
	~ClothSim();
	int go();
	
	Cloth* cloth;
	

	
	void init();
	void render();
	void update();
	void keypress(unsigned char key);
	void skeypress(int key);
	void mouseaction(int button);
	
private:
	
	static GLuint loadTexture();
	
	static void getTrianglePoints(int tri,int* x0,int* x1,int* x2,int size);
	
	
	GLushort indexArray[kNumberTriangles*2];
	GLushort text_verts[kNumberParticles*2];
	
	
	float viewer[3];
	float theta;
	float phi;
	float rho;
	float yoff;
	
	bool running;
	
	GLuint clothtex;
	
	GLfloat lightPos0[4];
	
	cudaGraphicsResource* positionsVBO_CUDA;
	
	GLuint positionsVBO;
	
	cudaGraphicsResource* normalsVBO_CUDA;
	
	GLuint normalsVBO;
};


static void tick();

static void display();

static void skeyboard(int key,int x, int y);

static void keyboard(unsigned char key, int x,int y);

static void mouse(int button, int state, int x, int y);

static void reshape (int w, int h); 

