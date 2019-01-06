#include "Sad.h"

void Sad::sadFace_1()
{
	//glClearColor(1, 1, 1, 1);
	Draw dukhi;
	glColor3f(0.137255f, 0.556863f, 1.0f);
	dukhi.drawCircle(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, 0, 120, 90, 0.867, 1);  //  face
	glColor3d(0, 0, 0);
	glColor3f(0.0f, 0.0f, 0.0f);
	dukhi.drawArc(SCREEN_WIDTH / 2 - 60, SCREEN_HEIGHT / 2 + 80, 0, 24, 259, 349, 50); // left eyebrow
	dukhi.drawArc(SCREEN_WIDTH / 2 + 65, SCREEN_HEIGHT / 2 + 80, 0, 24, 195, 285, 50); // right eyebrow

	dukhi.drawArc(SCREEN_WIDTH / 2 - 40, SCREEN_HEIGHT / 2 + 40, 0, 24, 195, 349, 50); // left eye
	dukhi.drawArc(SCREEN_WIDTH / 2 - 40, SCREEN_HEIGHT / 2 + 40, 0, 25, 195, 349, 50); // left eye
	dukhi.drawArc(SCREEN_WIDTH / 2 - 40, SCREEN_HEIGHT / 2 + 40, 0, 26, 195, 349, 50); // left eye
	dukhi.drawArc(SCREEN_WIDTH / 2 - 40, SCREEN_HEIGHT / 2 + 40, 0, 27, 195, 349, 50); // left eye
	dukhi.drawArc(SCREEN_WIDTH / 2 - 40, SCREEN_HEIGHT / 2 + 40, 0, 28, 195, 349, 50); // left eye
	dukhi.drawArc(SCREEN_WIDTH / 2 + 40, SCREEN_HEIGHT / 2 + 40, 0, 24, 195, 349, 50); // right eye
	dukhi.drawArc(SCREEN_WIDTH / 2 + 40, SCREEN_HEIGHT / 2 + 40, 0, 25, 195, 349, 50); // right eye
	dukhi.drawArc(SCREEN_WIDTH / 2 + 40, SCREEN_HEIGHT / 2 + 40, 0, 26, 195, 349, 50); // right eye
	dukhi.drawArc(SCREEN_WIDTH / 2 + 40, SCREEN_HEIGHT / 2 + 40, 0, 27, 195, 349, 50); // right eye
	dukhi.drawArc(SCREEN_WIDTH / 2 + 40, SCREEN_HEIGHT / 2 + 40, 0, 28, 195, 349, 50); // right eye

	dukhi.drawLine(SCREEN_WIDTH / 2 - 35, SCREEN_HEIGHT / 2 - 40, SCREEN_WIDTH / 2 + 40, SCREEN_HEIGHT / 2 - 40); // mouth 
	dukhi.drawLine(SCREEN_WIDTH / 2 - 35, SCREEN_HEIGHT / 2 - 41, SCREEN_WIDTH / 2 + 40, SCREEN_HEIGHT / 2 - 41); // mouth 
	dukhi.drawLine(SCREEN_WIDTH / 2 - 35, SCREEN_HEIGHT / 2 - 42, SCREEN_WIDTH / 2 + 40, SCREEN_HEIGHT / 2 - 42); // mouth 
	dukhi.drawLine(SCREEN_WIDTH / 2 - 35, SCREEN_HEIGHT / 2 - 43, SCREEN_WIDTH / 2 + 40, SCREEN_HEIGHT / 2 - 43); // mouth 
	dukhi.drawLine(SCREEN_WIDTH / 2 - 35, SCREEN_HEIGHT / 2 - 44, SCREEN_WIDTH / 2 + 40, SCREEN_HEIGHT / 2 - 44); // mouth 
	dukhi.drawLine(SCREEN_WIDTH / 2 - 35, SCREEN_HEIGHT / 2 - 45, SCREEN_WIDTH / 2 + 40, SCREEN_HEIGHT / 2 - 45); // mouth 

}

void Sad::sadFace_2(int offset)
{
	Draw dukhi;

	glColor3f(0.897f, 0.7895f, 0.12344f);
	//glColor3f(1.0f, 0.2588244f, 0.0f);
	dukhi.drawCircle(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2, 0, 120, 90, 1, 1);  //  face
	glColor3d(1, 1, 1);//white
	dukhi.drawCircle(SCREEN_WIDTH / 2 + 38 + offset, SCREEN_HEIGHT / 2 + 50, 0, 20, 100, 1, 1.5); // right
	dukhi.drawCircle(SCREEN_WIDTH / 2 - 40 + offset, SCREEN_HEIGHT / 2 + 50, 0, 20, 100, 1, 1.5); // left
	glColor3d(0, 0, 0);//black
	dukhi.drawCircle(SCREEN_WIDTH / 2 + 33 + offset, SCREEN_HEIGHT / 2 + 50, 0, 10, 100, 1, 1.5); // right
	dukhi.drawCircle(SCREEN_WIDTH / 2 - 35 + offset, SCREEN_HEIGHT / 2 + 50, 0, 10, 100, 1, 1.5); // left

	GLfloat r1 = 45, theta1 = 30, theta2 = 150, xc, yc;
	xc = SCREEN_WIDTH / 2.0f;
	yc = SCREEN_HEIGHT / 2.0f - 70.0f;

	dukhi.drawArc(xc + offset, yc, 0, r1, theta1, theta2, 50); //mouth
	dukhi.drawArc(xc + offset, yc, 0, r1 + 1, theta1, theta2, 50); //mouth
	dukhi.drawArc(xc + offset, yc, 0, r1 + 2, theta1, theta2, 50); //mouth
	dukhi.drawArc(xc + offset, yc, 0, r1 + 3, theta1, theta2, 50); //mouth
}

void Sad::sadFace_3(int offset)
{
	Draw dukhi;

	glColor3f(0.897f, 0.7895f, 0.12344f);
	dukhi.drawCircle(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2, 0, 120, 90, 0.867, 1);  //  face
	//glColor3d(1, 1, 1);//white
	glColor3d(0, 0, 0);
	dukhi.drawArc(SCREEN_WIDTH / 2 - 40 + offset, SCREEN_HEIGHT / 2 + 40, 0, 24, 195, 349, 50); // left eye
	dukhi.drawArc(SCREEN_WIDTH / 2 - 40 + offset, SCREEN_HEIGHT / 2 + 40, 0, 25, 195, 349, 50); // left eye
	dukhi.drawArc(SCREEN_WIDTH / 2 - 40 + offset, SCREEN_HEIGHT / 2 + 40, 0, 26, 195, 349, 50); // left eye
	dukhi.drawArc(SCREEN_WIDTH / 2 - 40 + offset, SCREEN_HEIGHT / 2 + 40, 0, 27, 195, 349, 50); // left eye
	dukhi.drawArc(SCREEN_WIDTH / 2 - 40 + offset, SCREEN_HEIGHT / 2 + 40, 0, 28, 195, 349, 50); // left eye
	dukhi.drawArc(SCREEN_WIDTH / 2 + 40 + offset, SCREEN_HEIGHT / 2 + 40, 0, 24, 195, 349, 50); // right eye
	dukhi.drawArc(SCREEN_WIDTH / 2 + 40 + offset, SCREEN_HEIGHT / 2 + 40, 0, 25, 195, 349, 50); // right eye
	dukhi.drawArc(SCREEN_WIDTH / 2 + 40 + offset, SCREEN_HEIGHT / 2 + 40, 0, 26, 195, 349, 50); // right eye
	dukhi.drawArc(SCREEN_WIDTH / 2 + 40 + offset, SCREEN_HEIGHT / 2 + 40, 0, 27, 195, 349, 50); // right eye
	dukhi.drawArc(SCREEN_WIDTH / 2 + 40 + offset, SCREEN_HEIGHT / 2 + 40, 0, 28, 195, 349, 50); // right eye

	dukhi.drawCircle(SCREEN_WIDTH / 2 + 38 + offset, SCREEN_HEIGHT / 2 + 50, 0, 20, 100, 1, 1); // right
	dukhi.drawCircle(SCREEN_WIDTH / 2 - 40 + offset, SCREEN_HEIGHT / 2 + 50, 0, 20, 100, 1, 1); // left
	glColor3b(1, 1, 1);
	//dukhi.drawLine(SCREEN_WIDTH / 2 - 40 + offset, SCREEN_HEIGHT / 2 - 40, SCREEN_WIDTH / 2 + 50 + offset, SCREEN_HEIGHT / 2 - 40); // mouth line-1 (+5,-40)
	dukhi.drawLine(SCREEN_WIDTH / 2 - 45 + offset, SCREEN_HEIGHT / 2 - 40, SCREEN_WIDTH / 2 - 22.5 + offset, SCREEN_HEIGHT / 2 - 10); // mouth line-1
	dukhi.drawLine(SCREEN_WIDTH / 2 - 45 + offset, SCREEN_HEIGHT / 2 - 41, SCREEN_WIDTH / 2 - 22.5 + offset, SCREEN_HEIGHT / 2 - 11); // mouth line-1
	dukhi.drawLine(SCREEN_WIDTH / 2 - 45 + offset, SCREEN_HEIGHT / 2 - 42, SCREEN_WIDTH / 2 - 22.5 + offset, SCREEN_HEIGHT / 2 - 12); // mouth line-1
	dukhi.drawLine(SCREEN_WIDTH / 2 - 45 + offset, SCREEN_HEIGHT / 2 - 43, SCREEN_WIDTH / 2 - 22.5 + offset, SCREEN_HEIGHT / 2 - 13); // mouth line-1

	dukhi.drawLine(SCREEN_WIDTH / 2 - 22.5 + offset, SCREEN_HEIGHT / 2 - 10, SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 40); // mouth line-2
	dukhi.drawLine(SCREEN_WIDTH / 2 - 22.5 + offset, SCREEN_HEIGHT / 2 - 11, SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 41); // mouth line-2
	dukhi.drawLine(SCREEN_WIDTH / 2 - 22.5 + offset, SCREEN_HEIGHT / 2 - 12, SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 42); // mouth line-2
	dukhi.drawLine(SCREEN_WIDTH / 2 - 22.5 + offset, SCREEN_HEIGHT / 2 - 13, SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 43); // mouth line-2

	dukhi.drawLine(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 40, SCREEN_WIDTH / 2 + 22.5 + offset, SCREEN_HEIGHT / 2 - 10); // mouth line-3
	dukhi.drawLine(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 41, SCREEN_WIDTH / 2 + 22.5 + offset, SCREEN_HEIGHT / 2 - 11); // mouth line-3
	dukhi.drawLine(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 42, SCREEN_WIDTH / 2 + 22.5 + offset, SCREEN_HEIGHT / 2 - 12); // mouth line-3
	dukhi.drawLine(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 43, SCREEN_WIDTH / 2 + 22.5 + offset, SCREEN_HEIGHT / 2 - 13); // mouth line-3

	dukhi.drawLine(SCREEN_WIDTH / 2 + 22.5 + offset, SCREEN_HEIGHT / 2 - 10, SCREEN_WIDTH / 2 + 45 + offset, SCREEN_HEIGHT / 2 - 40); // mouth line-4
	dukhi.drawLine(SCREEN_WIDTH / 2 + 22.5 + offset, SCREEN_HEIGHT / 2 - 11, SCREEN_WIDTH / 2 + 45 + offset, SCREEN_HEIGHT / 2 - 41); // mouth line-4
	dukhi.drawLine(SCREEN_WIDTH / 2 + 22.5 + offset, SCREEN_HEIGHT / 2 - 12, SCREEN_WIDTH / 2 + 45 + offset, SCREEN_HEIGHT / 2 - 42); // mouth line-4
	dukhi.drawLine(SCREEN_WIDTH / 2 + 22.5 + offset, SCREEN_HEIGHT / 2 - 13, SCREEN_WIDTH / 2 + 45 + offset, SCREEN_HEIGHT / 2 - 43); // mouth line-4
}

Sad::Sad()
{
}


Sad::~Sad()
{
}
