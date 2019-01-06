#ifndef DRAW_1
#define DRAW_1

#include <iostream>
//#include "pch.h"
#include <GL/glew.h> 
#include <GLFW/glfw3.h>
#include <math.h>
using namespace std;

#define SCREEN_WIDTH 640
#define SCREEN_HEIGHT 480
#define M_PI 3.14159265
#define RadToDeg(rad) (180 / M_PI)*rad
#define DegToRad(deg) (M_PI / 180)*deg

class Draw {

public:
	Draw();
	void drawCircle(GLfloat x, GLfloat y, GLfloat z, GLfloat radius, GLint numberOfSides, GLfloat factorx, GLfloat factory);
	void drawCircleHollow(GLfloat x, GLfloat y, GLfloat z, GLfloat radius, GLint numberOfSides);
	void drawArc(GLfloat x, GLfloat y, GLfloat z, GLfloat radius, int start, int end, GLint numberOfSides);
	void drawLine(GLfloat x1, GLfloat y1, GLfloat x2, GLfloat y2);
	void drawSemiCircle(GLfloat x, GLfloat y, GLfloat z, GLfloat radius, GLint numberOfSides, GLfloat factorx, GLfloat factory);
	~Draw();

};

#endif // !DRAW_1

