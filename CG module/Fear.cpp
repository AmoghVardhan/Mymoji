#include "Fear.h"

void Fear::fearFace_1()
{
	Draw darr;

	glColor3f(0.897f, 0.7895f, 0.12344f);
	//glColor3f(1.0f, 0.2588244f, 0.0f);
	darr.drawCircle(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, 0, 120, 90, 1, 1);  //  face
	glColor3d(1, 1, 1);//white
	darr.drawCircle(SCREEN_WIDTH / 2 + 38, SCREEN_HEIGHT / 2 + 50, 0, 20, 100, 1, 1); // right
	darr.drawCircle(SCREEN_WIDTH / 2 - 40, SCREEN_HEIGHT / 2 + 50, 0, 20, 100, 1, 1); // left
	glColor3d(0, 0, 0);
	darr.drawCircle(SCREEN_WIDTH / 2 + 38, SCREEN_HEIGHT / 2 + 50, 0, 10, 100, 1, 1); // right
	darr.drawCircle(SCREEN_WIDTH / 2 - 40, SCREEN_HEIGHT / 2 + 50, 0, 10, 100, 1, 1); // left
	glColor3f(0.0f, 0.0f, 0.0f);
	darr.drawArc(SCREEN_WIDTH / 2 - 45, SCREEN_HEIGHT / 2 + 50, 0, 30, 50, 120, 50); // left eyebrow
	darr.drawArc(SCREEN_WIDTH / 2 - 45, SCREEN_HEIGHT / 2 + 50, 0, 31, 50, 120, 50); // left eyebrow
	darr.drawArc(SCREEN_WIDTH / 2 - 45, SCREEN_HEIGHT / 2 + 50, 0, 32, 50, 120, 50); // left eyebrow
	darr.drawArc(SCREEN_WIDTH / 2 - 45, SCREEN_HEIGHT / 2 + 50, 0, 33, 50, 120, 50); // left eyebrow
	darr.drawArc(SCREEN_WIDTH / 2 + 40, SCREEN_HEIGHT / 2 + 50, 0, 30, 60, 130, 50); // right eyebrow
	darr.drawArc(SCREEN_WIDTH / 2 + 40, SCREEN_HEIGHT / 2 + 50, 0, 31, 60, 130, 50); // right eyebrow
	darr.drawArc(SCREEN_WIDTH / 2 + 40, SCREEN_HEIGHT / 2 + 50, 0, 32, 60, 130, 50); // right eyebrow
	darr.drawArc(SCREEN_WIDTH / 2 + 40, SCREEN_HEIGHT / 2 + 50, 0, 33, 60, 130, 50); // right eyebrow

	GLfloat r1 = 45, theta1 = 30, theta2 = 150, xc, yc;
	xc = SCREEN_WIDTH / 2.0f;
	yc = SCREEN_HEIGHT / 2.0f - 70.0f;

	glColor3f(0.46f, 0.25f, 0.20f);
	//glColor3d(128, 0, 0);
	darr.drawSemiCircle(xc, yc, 0, 40, 100, 0.9, 2.5); // mouth 
	glColor3f(1.0f, 0.0f, 0.0f);
	darr.drawSemiCircle(xc, yc, 0, 30, 100, 1, 1); // mouth 
	//darr.drawArc(xc, yc, 0, r1, theta1, theta2, 50); //mouth
}


void Fear::fearFace_2(int offset)
{
	Draw darr;

	glColor3f(0.897f, 0.7895f, 0.12344f);
	darr.drawCircle(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2, 0, 120, 90, 1, 1);  //  face
	//glColor3d(1, 1, 1);//white
	glColor3d(0, 0, 0);
	darr.drawCircle(SCREEN_WIDTH / 2 + 38 + offset, SCREEN_HEIGHT / 2 + 50, 0, 20, 100, 1, 1); // right
	darr.drawCircle(SCREEN_WIDTH / 2 - 40 + offset, SCREEN_HEIGHT / 2 + 50, 0, 20, 100, 1, 1); // left
	glColor3f(0.897f, 0.7895f, 0.12344f);
	darr.drawCircle(SCREEN_WIDTH / 2 + 38 + offset, SCREEN_HEIGHT / 2 + 50, 0, 10, 100, 1, 1); // right
	darr.drawCircle(SCREEN_WIDTH / 2 - 40 + offset, SCREEN_HEIGHT / 2 + 50, 0, 10, 100, 1, 1); // left
	glColor3f(0.0f, 0.0f, 0.0f);//black

	darr.drawLine(SCREEN_WIDTH / 2 - 70 + offset, SCREEN_HEIGHT / 2 + 65, SCREEN_WIDTH / 2 - 45 + offset, SCREEN_HEIGHT / 2 + 85); // left eyebrow
	darr.drawLine(SCREEN_WIDTH / 2 - 70 + offset, SCREEN_HEIGHT / 2 + 64, SCREEN_WIDTH / 2 - 45 + offset, SCREEN_HEIGHT / 2 + 84); // left eyebrow
	darr.drawLine(SCREEN_WIDTH / 2 - 70 + offset, SCREEN_HEIGHT / 2 + 63, SCREEN_WIDTH / 2 - 45 + offset, SCREEN_HEIGHT / 2 + 83); // left eyebrow
	darr.drawLine(SCREEN_WIDTH / 2 - 70 + offset, SCREEN_HEIGHT / 2 + 62, SCREEN_WIDTH / 2 - 45 + offset, SCREEN_HEIGHT / 2 + 82); // left eyebrow

	darr.drawLine(SCREEN_WIDTH / 2 + 63 + offset, SCREEN_HEIGHT / 2 + 70, SCREEN_WIDTH / 2 + 33 + offset, SCREEN_HEIGHT / 2 + 85); // right eyebrow
	darr.drawLine(SCREEN_WIDTH / 2 + 63 + offset, SCREEN_HEIGHT / 2 + 69, SCREEN_WIDTH / 2 + 33 + offset, SCREEN_HEIGHT / 2 + 84); // right eyebrow
	darr.drawLine(SCREEN_WIDTH / 2 + 63 + offset, SCREEN_HEIGHT / 2 + 68, SCREEN_WIDTH / 2 + 33 + offset, SCREEN_HEIGHT / 2 + 83); // right eyebrow
	darr.drawLine(SCREEN_WIDTH / 2 + 63 + offset, SCREEN_HEIGHT / 2 + 67, SCREEN_WIDTH / 2 + 33 + offset, SCREEN_HEIGHT / 2 + 82); // right eyebrow

	glColor3f(1.0f, 1.0f, 1.0f); // white

	darr.drawLine(SCREEN_WIDTH / 2 - 30 + offset, SCREEN_HEIGHT / 2 - 50, SCREEN_WIDTH / 2 + 30 + offset, SCREEN_HEIGHT / 2 - 30); // mouth 
	darr.drawLine(SCREEN_WIDTH / 2 - 30 + offset, SCREEN_HEIGHT / 2 - 49, SCREEN_WIDTH / 2 + 30 + offset, SCREEN_HEIGHT / 2 - 29); // mouth
	darr.drawLine(SCREEN_WIDTH / 2 - 30 + offset, SCREEN_HEIGHT / 2 - 48, SCREEN_WIDTH / 2 + 30 + offset, SCREEN_HEIGHT / 2 - 28); // mouth
	darr.drawLine(SCREEN_WIDTH / 2 - 30 + offset, SCREEN_HEIGHT / 2 - 47, SCREEN_WIDTH / 2 + 30 + offset, SCREEN_HEIGHT / 2 - 27); // mouth
	darr.drawLine(SCREEN_WIDTH / 2 - 30 + offset, SCREEN_HEIGHT / 2 - 46, SCREEN_WIDTH / 2 + 30 + offset, SCREEN_HEIGHT / 2 - 26); // mouth
	darr.drawLine(SCREEN_WIDTH / 2 - 30 + offset, SCREEN_HEIGHT / 2 - 45, SCREEN_WIDTH / 2 + 30 + offset, SCREEN_HEIGHT / 2 - 25); // mouth
	//darr.drawLine(SCREEN_WIDTH / 2 - 30 + offset, SCREEN_HEIGHT / 2 - 44, SCREEN_WIDTH / 2 + 30 + offset, SCREEN_HEIGHT / 2 - 24); // mouth
	//darr.drawLine(SCREEN_WIDTH / 2 - 30 + offset, SCREEN_HEIGHT / 2 - 43, SCREEN_WIDTH / 2 + 30 + offset, SCREEN_HEIGHT / 2 - 23); // mouth
}

void Fear::fearFace_3(int offset)
{
	Draw darr;

	glColor3f(0.897f, 0.7895f, 0.12344f);
	darr.drawCircle(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2, 0, 120, 90, 1, 1);  //  face factorx = 0.867
	//glColor3d(1, 1, 1);//white
	glColor3d(0, 0, 0);
	darr.drawArc(SCREEN_WIDTH / 2 - 80 + offset, SCREEN_HEIGHT / 2 + 90, 0, 40, 10, -90, 50); // left eyebrow
	darr.drawArc(SCREEN_WIDTH / 2 - 80 + offset, SCREEN_HEIGHT / 2 + 90, 0, 39, 10, -90, 50); // left eyebrow
	darr.drawArc(SCREEN_WIDTH / 2 - 80 + offset, SCREEN_HEIGHT / 2 + 90, 0, 38, 10, -90, 50); // left eyebrow
	darr.drawArc(SCREEN_WIDTH / 2 - 80 + offset, SCREEN_HEIGHT / 2 + 90, 0, 37, 10, -90, 50); // left eyebrow
	
	darr.drawArc(SCREEN_WIDTH / 2 + 80 + offset, SCREEN_HEIGHT / 2 + 90, 0, 40, 170, 260, 50); // right eyebrow
	darr.drawArc(SCREEN_WIDTH / 2 + 80 + offset, SCREEN_HEIGHT / 2 + 90, 0, 41, 170, 260, 50); // right eyebrow
	darr.drawArc(SCREEN_WIDTH / 2 + 80 + offset, SCREEN_HEIGHT / 2 + 90, 0, 42, 170, 260, 50); // right eyebrow
	darr.drawArc(SCREEN_WIDTH / 2 + 80 + offset, SCREEN_HEIGHT / 2 + 90, 0, 43, 170, 260, 50); // right eyebrow
	
	glColor3d(0, 0, 0);
	darr.drawCircle(SCREEN_WIDTH / 2 + 38 + offset, SCREEN_HEIGHT / 2 + 50, 0, 20, 100, 0.7, 1); // right
	darr.drawCircle(SCREEN_WIDTH / 2 - 40 + offset, SCREEN_HEIGHT / 2 + 50, 0, 20, 100, 0.7, 1); // left
	glColor3b(1, 1, 1);
	//darr.drawLine(SCREEN_WIDTH / 2 - 40 + offset, SCREEN_HEIGHT / 2 - 40, SCREEN_WIDTH / 2 + 50 + offset, SCREEN_HEIGHT / 2 - 40); // mouth line-1 (+5,-40)
	darr.drawLine(SCREEN_WIDTH / 2 - 45 + offset, SCREEN_HEIGHT / 2 - 40, SCREEN_WIDTH / 2 - 22.5 + offset, SCREEN_HEIGHT / 2 - 10); // mouth line-1
	darr.drawLine(SCREEN_WIDTH / 2 - 45 + offset, SCREEN_HEIGHT / 2 - 41, SCREEN_WIDTH / 2 - 22.5 + offset, SCREEN_HEIGHT / 2 - 11); // mouth line-1
	darr.drawLine(SCREEN_WIDTH / 2 - 45 + offset, SCREEN_HEIGHT / 2 - 42, SCREEN_WIDTH / 2 - 22.5 + offset, SCREEN_HEIGHT / 2 - 12); // mouth line-1
	darr.drawLine(SCREEN_WIDTH / 2 - 45 + offset, SCREEN_HEIGHT / 2 - 43, SCREEN_WIDTH / 2 - 22.5 + offset, SCREEN_HEIGHT / 2 - 13); // mouth line-1
	darr.drawLine(SCREEN_WIDTH / 2 - 45 + offset, SCREEN_HEIGHT / 2 - 42, SCREEN_WIDTH / 2 - 22.5 + offset, SCREEN_HEIGHT / 2 - 12); // mouth line-1
	darr.drawLine(SCREEN_WIDTH / 2 - 45 + offset, SCREEN_HEIGHT / 2 - 41, SCREEN_WIDTH / 2 - 22.5 + offset, SCREEN_HEIGHT / 2 - 11); // mouth line-1

	darr.drawLine(SCREEN_WIDTH / 2 - 22.5 + offset, SCREEN_HEIGHT / 2 - 10, SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 40); // mouth line-2
	darr.drawLine(SCREEN_WIDTH / 2 - 22.5 + offset, SCREEN_HEIGHT / 2 - 11, SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 41); // mouth line-2
	darr.drawLine(SCREEN_WIDTH / 2 - 22.5 + offset, SCREEN_HEIGHT / 2 - 12, SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 42); // mouth line-2
	darr.drawLine(SCREEN_WIDTH / 2 - 22.5 + offset, SCREEN_HEIGHT / 2 - 13, SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 43); // mouth line-2
	darr.drawLine(SCREEN_WIDTH / 2 - 22.5 + offset, SCREEN_HEIGHT / 2 - 14, SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 42); // mouth line-2
	darr.drawLine(SCREEN_WIDTH / 2 - 22.5 + offset, SCREEN_HEIGHT / 2 - 15, SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 41); // mouth line-2

	darr.drawLine(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 40, SCREEN_WIDTH / 2 + 22.5 + offset, SCREEN_HEIGHT / 2 - 10); // mouth line-3
	darr.drawLine(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 41, SCREEN_WIDTH / 2 + 22.5 + offset, SCREEN_HEIGHT / 2 - 11); // mouth line-3
	darr.drawLine(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 42, SCREEN_WIDTH / 2 + 22.5 + offset, SCREEN_HEIGHT / 2 - 12); // mouth line-3
	darr.drawLine(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 43, SCREEN_WIDTH / 2 + 22.5 + offset, SCREEN_HEIGHT / 2 - 13); // mouth line-3
	darr.drawLine(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 42, SCREEN_WIDTH / 2 + 22.5 + offset, SCREEN_HEIGHT / 2 - 12); // mouth line-3
	darr.drawLine(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 41, SCREEN_WIDTH / 2 + 22.5 + offset, SCREEN_HEIGHT / 2 - 11); // mouth line-3

	darr.drawLine(SCREEN_WIDTH / 2 + 22.5 + offset, SCREEN_HEIGHT / 2 - 10, SCREEN_WIDTH / 2 + 45 + offset, SCREEN_HEIGHT / 2 - 40); // mouth line-4
	darr.drawLine(SCREEN_WIDTH / 2 + 22.5 + offset, SCREEN_HEIGHT / 2 - 11, SCREEN_WIDTH / 2 + 45 + offset, SCREEN_HEIGHT / 2 - 41); // mouth line-4
	darr.drawLine(SCREEN_WIDTH / 2 + 22.5 + offset, SCREEN_HEIGHT / 2 - 12, SCREEN_WIDTH / 2 + 45 + offset, SCREEN_HEIGHT / 2 - 42); // mouth line-4
	darr.drawLine(SCREEN_WIDTH / 2 + 22.5 + offset, SCREEN_HEIGHT / 2 - 13, SCREEN_WIDTH / 2 + 45 + offset, SCREEN_HEIGHT / 2 - 43); // mouth line-4
	darr.drawLine(SCREEN_WIDTH / 2 + 22.5 + offset, SCREEN_HEIGHT / 2 - 14, SCREEN_WIDTH / 2 + 45 + offset, SCREEN_HEIGHT / 2 - 44); // mouth line-4
	darr.drawLine(SCREEN_WIDTH / 2 + 22.5 + offset, SCREEN_HEIGHT / 2 - 15, SCREEN_WIDTH / 2 + 45 + offset, SCREEN_HEIGHT / 2 - 45); // mouth line-4
}


Fear::Fear()
{
}


Fear::~Fear()
{
}
