

#include "constants.h"





class Cloth
{
public:
	Cloth(float* x);
	~Cloth();
	void stepForeward(float* x);
	void toggleWind();
private:
	
	State* state;
	bool windon;
};

