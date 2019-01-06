#include "Disgust.h"


void Disgust::disgustface_1()
{
		Draw ghrna;

		//glColor3f(0.897f, 0.7895f, 0.12344f);
		glColor3f(0.419608f, 0.556863f, 0.137255);
		ghrna.drawCircle(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, 0, 120, 90, 0.867, 1);  //  face
		glColor3d(0, 0, 0);
		ghrna.drawCircle(SCREEN_WIDTH / 2 + 38, SCREEN_HEIGHT / 2 + 50, 0, 15, 100, 0.7, 1); // right
		ghrna.drawCircle(SCREEN_WIDTH / 2 - 40, SCREEN_HEIGHT / 2 + 50, 0, 15, 100, 0.7, 1); // left
		glColor3f(0.0f, 0.0f, 0.0f);
		ghrna.drawArc(SCREEN_WIDTH / 2 - 60, SCREEN_HEIGHT / 2 + 90, 0, 24, 259, 349, 50); // left eyebrow
		ghrna.drawArc(SCREEN_WIDTH / 2 + 55, SCREEN_HEIGHT / 2 + 90, 0, 24, 195, 285, 50); // right eyebrow

		GLfloat r1 = 45, theta1 = 30, theta2 = 150, xc, yc, x1, x2, y1, y2, xcl, ycl, xcr, ycr, rr, rl;
		xc = SCREEN_WIDTH / 2.0f;
		yc = SCREEN_HEIGHT / 2.0f - 40.0f;

		x1 = xc + r1 * cos(DegToRad(theta1));
		y1 = yc + r1 * sin(DegToRad(theta1));
		x2 = xc + r1 * cos(DegToRad(theta2));
		y2 = yc + r1 * sin(DegToRad(theta2));

		xcr = xc + 80;
		xcl = xc - 80;
		ycr = yc;
		ycl = yc;

		rr = sqrt(pow(x1 - xcr, 2) + pow(y1 - ycr, 2));
		rl = sqrt(pow(x2 - xcl, 2) + pow(y2 - ycl, 2));

		ghrna.drawArc(xc, yc, 0, r1, theta1, theta2, 50); //mouth
		ghrna.drawArc(xcl, ycl, 0, rl, -01, 60, 50);//mouth left
		ghrna.drawArc(xcr, ycr, 0, rr, 120, 180, 50);//mouth left
}

void Disgust::disgustface_2(int offset)
{
	Draw ghrna;

	glColor3d(1, 1, 1);
	ghrna.drawCircle(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2, 0, 130, 90, 1, 1);  //  face
	glColor3f(0.897f, 0.7895f, 0.12344f);
	ghrna.drawCircle(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2, 0, 120, 90, 1, 1);  //  face
	glColor3d(0, 0, 0);

	ghrna.drawLine(SCREEN_WIDTH / 2 - 55 + offset, SCREEN_HEIGHT / 2 + 40, SCREEN_WIDTH / 2 - 25 + offset, SCREEN_HEIGHT / 2 + 35); // left eyebrow-1
	ghrna.drawLine(SCREEN_WIDTH / 2 - 55 + offset, SCREEN_HEIGHT / 2 + 41, SCREEN_WIDTH / 2 - 25 + offset, SCREEN_HEIGHT / 2 + 36); // left eyebrow-1
	ghrna.drawLine(SCREEN_WIDTH / 2 - 55 + offset, SCREEN_HEIGHT / 2 + 42, SCREEN_WIDTH / 2 - 25 + offset, SCREEN_HEIGHT / 2 + 37); // left eyebrow-1
	ghrna.drawLine(SCREEN_WIDTH / 2 - 55 + offset, SCREEN_HEIGHT / 2 + 43, SCREEN_WIDTH / 2 - 25 + offset, SCREEN_HEIGHT / 2 + 38); // left eyebrow-1

	ghrna.drawLine(SCREEN_WIDTH / 2 - 25 + offset, SCREEN_HEIGHT / 2 + 35, SCREEN_WIDTH / 2 - 55 + offset, SCREEN_HEIGHT / 2 + 30); // left eyebrow-2
	ghrna.drawLine(SCREEN_WIDTH / 2 - 25 + offset, SCREEN_HEIGHT / 2 + 36, SCREEN_WIDTH / 2 - 55 + offset, SCREEN_HEIGHT / 2 + 31); // left eyebrow-2
	ghrna.drawLine(SCREEN_WIDTH / 2 - 25 + offset, SCREEN_HEIGHT / 2 + 37, SCREEN_WIDTH / 2 - 55 + offset, SCREEN_HEIGHT / 2 + 32); // left eyebrow-2
	ghrna.drawLine(SCREEN_WIDTH / 2 - 25 + offset, SCREEN_HEIGHT / 2 + 38, SCREEN_WIDTH / 2 - 55 + offset, SCREEN_HEIGHT / 2 + 33); // left eyebrow-2

	ghrna.drawLine(SCREEN_WIDTH / 2 + 55 + offset, SCREEN_HEIGHT / 2 + 40, SCREEN_WIDTH / 2 + 25 + offset, SCREEN_HEIGHT / 2 + 35); // right eyebrow-1
	ghrna.drawLine(SCREEN_WIDTH / 2 + 55 + offset, SCREEN_HEIGHT / 2 + 41, SCREEN_WIDTH / 2 + 25 + offset, SCREEN_HEIGHT / 2 + 36); // right eyebrow-1
	ghrna.drawLine(SCREEN_WIDTH / 2 + 55 + offset, SCREEN_HEIGHT / 2 + 42, SCREEN_WIDTH / 2 + 25 + offset, SCREEN_HEIGHT / 2 + 37); // right eyebrow-1
	ghrna.drawLine(SCREEN_WIDTH / 2 + 55 + offset, SCREEN_HEIGHT / 2 + 43, SCREEN_WIDTH / 2 + 25 + offset, SCREEN_HEIGHT / 2 + 38); // right eyebrow-1

	ghrna.drawLine(SCREEN_WIDTH / 2 + 25 + offset, SCREEN_HEIGHT / 2 + 35, SCREEN_WIDTH / 2 + 55 + offset, SCREEN_HEIGHT / 2 + 30); // right eyebrow-2
	ghrna.drawLine(SCREEN_WIDTH / 2 + 25 + offset, SCREEN_HEIGHT / 2 + 36, SCREEN_WIDTH / 2 + 55 + offset, SCREEN_HEIGHT / 2 + 31); // right eyebrow-2
	ghrna.drawLine(SCREEN_WIDTH / 2 + 25 + offset, SCREEN_HEIGHT / 2 + 37, SCREEN_WIDTH / 2 + 55 + offset, SCREEN_HEIGHT / 2 + 32); // right eyebrow-2
	ghrna.drawLine(SCREEN_WIDTH / 2 + 25 + offset, SCREEN_HEIGHT / 2 + 38, SCREEN_WIDTH / 2 + 55 + offset, SCREEN_HEIGHT / 2 + 33); // right eyebrow-2

	//ghrna.drawLine(SCREEN_WIDTH / 2 - 40 + offset, SCREEN_HEIGHT / 2 - 20, SCREEN_WIDTH / 2 + 40 + offset, SCREEN_HEIGHT / 2 - 20); // mouth line

	ghrna.drawLine(SCREEN_WIDTH / 2 - 40 + offset, SCREEN_HEIGHT / 2 - 20, SCREEN_WIDTH / 2 - 30 + offset, SCREEN_HEIGHT / 2 - 10); // mouth line-1
	ghrna.drawLine(SCREEN_WIDTH / 2 - 40 + offset, SCREEN_HEIGHT / 2 - 21, SCREEN_WIDTH / 2 - 30 + offset, SCREEN_HEIGHT / 2 - 11); // mouth line-1
	ghrna.drawLine(SCREEN_WIDTH / 2 - 40 + offset, SCREEN_HEIGHT / 2 - 22, SCREEN_WIDTH / 2 - 30 + offset, SCREEN_HEIGHT / 2 - 12); // mouth line-1
	ghrna.drawLine(SCREEN_WIDTH / 2 - 40 + offset, SCREEN_HEIGHT / 2 - 23, SCREEN_WIDTH / 2 - 30 + offset, SCREEN_HEIGHT / 2 - 13); // mouth line-1

	ghrna.drawLine(SCREEN_WIDTH / 2 - 30 + offset, SCREEN_HEIGHT / 2 - 10, SCREEN_WIDTH / 2 - 20 + offset, SCREEN_HEIGHT / 2 - 20); // mouth line-2
	ghrna.drawLine(SCREEN_WIDTH / 2 - 30 + offset, SCREEN_HEIGHT / 2 - 11, SCREEN_WIDTH / 2 - 20 + offset, SCREEN_HEIGHT / 2 - 21); // mouth line-2
	ghrna.drawLine(SCREEN_WIDTH / 2 - 30 + offset, SCREEN_HEIGHT / 2 - 12, SCREEN_WIDTH / 2 - 20 + offset, SCREEN_HEIGHT / 2 - 22); // mouth line-2
	ghrna.drawLine(SCREEN_WIDTH / 2 - 30 + offset, SCREEN_HEIGHT / 2 - 13, SCREEN_WIDTH / 2 - 20 + offset, SCREEN_HEIGHT / 2 - 23); // mouth line-2

	ghrna.drawLine(SCREEN_WIDTH / 2 - 20 + offset, SCREEN_HEIGHT / 2 - 20, SCREEN_WIDTH / 2 - 10 + offset, SCREEN_HEIGHT / 2 - 10); // mouth line-3
	ghrna.drawLine(SCREEN_WIDTH / 2 - 20 + offset, SCREEN_HEIGHT / 2 - 21, SCREEN_WIDTH / 2 - 10 + offset, SCREEN_HEIGHT / 2 - 11); // mouth line-3
	ghrna.drawLine(SCREEN_WIDTH / 2 - 20 + offset, SCREEN_HEIGHT / 2 - 22, SCREEN_WIDTH / 2 - 10 + offset, SCREEN_HEIGHT / 2 - 12); // mouth line-3
	ghrna.drawLine(SCREEN_WIDTH / 2 - 20 + offset, SCREEN_HEIGHT / 2 - 23, SCREEN_WIDTH / 2 - 10 + offset, SCREEN_HEIGHT / 2 - 13); // mouth line-3

	ghrna.drawLine(SCREEN_WIDTH / 2 - 10 + offset, SCREEN_HEIGHT / 2 - 10, SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 20); // mouth line-4
	ghrna.drawLine(SCREEN_WIDTH / 2 - 10 + offset, SCREEN_HEIGHT / 2 - 11, SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 21); // mouth line-4
	ghrna.drawLine(SCREEN_WIDTH / 2 - 10 + offset, SCREEN_HEIGHT / 2 - 12, SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 22); // mouth line-4
	ghrna.drawLine(SCREEN_WIDTH / 2 - 10 + offset, SCREEN_HEIGHT / 2 - 13, SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 23); // mouth line-4

	ghrna.drawLine(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 20, SCREEN_WIDTH / 2 + 10 + offset, SCREEN_HEIGHT / 2 - 10); // mouth line-5
	ghrna.drawLine(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 21, SCREEN_WIDTH / 2 + 10 + offset, SCREEN_HEIGHT / 2 - 11); // mouth line-5
	ghrna.drawLine(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 22, SCREEN_WIDTH / 2 + 10 + offset, SCREEN_HEIGHT / 2 - 12); // mouth line-5
	ghrna.drawLine(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 23, SCREEN_WIDTH / 2 + 10 + offset, SCREEN_HEIGHT / 2 - 13); // mouth line-5

	ghrna.drawLine(SCREEN_WIDTH / 2 + 10 + offset, SCREEN_HEIGHT / 2 - 10, SCREEN_WIDTH / 2 + 20 + offset, SCREEN_HEIGHT / 2 - 20); // mouth line-6
	ghrna.drawLine(SCREEN_WIDTH / 2 + 10 + offset, SCREEN_HEIGHT / 2 - 11, SCREEN_WIDTH / 2 + 20 + offset, SCREEN_HEIGHT / 2 - 21); // mouth line-6
	ghrna.drawLine(SCREEN_WIDTH / 2 + 10 + offset, SCREEN_HEIGHT / 2 - 12, SCREEN_WIDTH / 2 + 20 + offset, SCREEN_HEIGHT / 2 - 22); // mouth line-6
	ghrna.drawLine(SCREEN_WIDTH / 2 + 10 + offset, SCREEN_HEIGHT / 2 - 13, SCREEN_WIDTH / 2 + 20 + offset, SCREEN_HEIGHT / 2 - 23); // mouth line-6

	ghrna.drawLine(SCREEN_WIDTH / 2 + 20 + offset, SCREEN_HEIGHT / 2 - 20, SCREEN_WIDTH / 2 + 30 + offset, SCREEN_HEIGHT / 2 - 10); // mouth line-7
	ghrna.drawLine(SCREEN_WIDTH / 2 + 20 + offset, SCREEN_HEIGHT / 2 - 21, SCREEN_WIDTH / 2 + 30 + offset, SCREEN_HEIGHT / 2 - 11); // mouth line-7
	ghrna.drawLine(SCREEN_WIDTH / 2 + 20 + offset, SCREEN_HEIGHT / 2 - 22, SCREEN_WIDTH / 2 + 30 + offset, SCREEN_HEIGHT / 2 - 12); // mouth line-7
	ghrna.drawLine(SCREEN_WIDTH / 2 + 20 + offset, SCREEN_HEIGHT / 2 - 23, SCREEN_WIDTH / 2 + 30 + offset, SCREEN_HEIGHT / 2 - 13); // mouth line-7

	ghrna.drawLine(SCREEN_WIDTH / 2 + 30 + offset, SCREEN_HEIGHT / 2 - 10, SCREEN_WIDTH / 2 + 40 + offset, SCREEN_HEIGHT / 2 - 20); // mouth line-8
	ghrna.drawLine(SCREEN_WIDTH / 2 + 30 + offset, SCREEN_HEIGHT / 2 - 11, SCREEN_WIDTH / 2 + 40 + offset, SCREEN_HEIGHT / 2 - 21); // mouth line-8
	ghrna.drawLine(SCREEN_WIDTH / 2 + 30 + offset, SCREEN_HEIGHT / 2 - 12, SCREEN_WIDTH / 2 + 40 + offset, SCREEN_HEIGHT / 2 - 22); // mouth line-8
	ghrna.drawLine(SCREEN_WIDTH / 2 + 30 + offset, SCREEN_HEIGHT / 2 - 13, SCREEN_WIDTH / 2 + 40 + offset, SCREEN_HEIGHT / 2 - 23); // mouth line-8

}

void Disgust::disgustface_3(int offset)
{
	Draw ghrna;

	glColor3f(0.5f, 0.5f, 0.5f);//grey
	ghrna.drawCircle(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2, 0, 130, 90, 1, 1);  //  face
	glColor3d(1, 1, 1);//white
	ghrna.drawCircle(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2, 0, 120, 90, 1, 1);  //  face
	glColor3d(0, 0, 0);//black
	ghrna.drawCircle(SCREEN_WIDTH / 2 - 60 + offset, SCREEN_HEIGHT / 2 + 30, 0, 40, 90, 1, 1); // left outer eye
	ghrna.drawCircle(SCREEN_WIDTH / 2 + 60 + offset, SCREEN_HEIGHT / 2 + 30, 0, 40, 90, 1, 1); // right outer eye

	glColor3d(1, 1, 1);//white
	ghrna.drawCircle(SCREEN_WIDTH / 2 - 60 + offset, SCREEN_HEIGHT / 2 + 30, 0, 30, 90, 1, 1); // left outer eye
	ghrna.drawCircle(SCREEN_WIDTH / 2 + 60 + offset, SCREEN_HEIGHT / 2 + 30, 0, 30, 90, 1, 1); // right outer eye

	glColor3d(0, 0, 0);//black
	ghrna.drawCircle(SCREEN_WIDTH / 2 - 60 + offset, SCREEN_HEIGHT / 2 + 30, 0, 10, 90, 1, 1); // left outer eye
	ghrna.drawCircle(SCREEN_WIDTH / 2 + 60 + offset, SCREEN_HEIGHT / 2 + 30, 0, 10, 90, 1, 1); // right outer eye

	ghrna.drawLine(SCREEN_WIDTH / 2 - 60 + offset, SCREEN_HEIGHT / 2 - 40, SCREEN_WIDTH / 2 - 45 + offset, SCREEN_HEIGHT / 2 - 30); // mouth line-1
	ghrna.drawLine(SCREEN_WIDTH / 2 - 60 + offset, SCREEN_HEIGHT / 2 - 41, SCREEN_WIDTH / 2 - 45 + offset, SCREEN_HEIGHT / 2 - 31); // mouth line-1
	ghrna.drawLine(SCREEN_WIDTH / 2 - 60 + offset, SCREEN_HEIGHT / 2 - 42, SCREEN_WIDTH / 2 - 45 + offset, SCREEN_HEIGHT / 2 - 32); // mouth line-1
	ghrna.drawLine(SCREEN_WIDTH / 2 - 60 + offset, SCREEN_HEIGHT / 2 - 43, SCREEN_WIDTH / 2 - 45 + offset, SCREEN_HEIGHT / 2 - 33); // mouth line-1

	ghrna.drawLine(SCREEN_WIDTH / 2 - 45 + offset, SCREEN_HEIGHT / 2 - 30, SCREEN_WIDTH / 2 - 30 + offset, SCREEN_HEIGHT / 2 - 40); // mouth line-2
	ghrna.drawLine(SCREEN_WIDTH / 2 - 45 + offset, SCREEN_HEIGHT / 2 - 31, SCREEN_WIDTH / 2 - 30 + offset, SCREEN_HEIGHT / 2 - 41); // mouth line-2
	ghrna.drawLine(SCREEN_WIDTH / 2 - 45 + offset, SCREEN_HEIGHT / 2 - 32, SCREEN_WIDTH / 2 - 30 + offset, SCREEN_HEIGHT / 2 - 42); // mouth line-2
	ghrna.drawLine(SCREEN_WIDTH / 2 - 45 + offset, SCREEN_HEIGHT / 2 - 33, SCREEN_WIDTH / 2 - 30 + offset, SCREEN_HEIGHT / 2 - 43); // mouth line-2

	ghrna.drawLine(SCREEN_WIDTH / 2 - 30 + offset, SCREEN_HEIGHT / 2 - 40, SCREEN_WIDTH / 2 - 15 + offset, SCREEN_HEIGHT / 2 - 30); // mouth line-3
	ghrna.drawLine(SCREEN_WIDTH / 2 - 30 + offset, SCREEN_HEIGHT / 2 - 41, SCREEN_WIDTH / 2 - 15 + offset, SCREEN_HEIGHT / 2 - 31); // mouth line-3
	ghrna.drawLine(SCREEN_WIDTH / 2 - 30 + offset, SCREEN_HEIGHT / 2 - 42, SCREEN_WIDTH / 2 - 15 + offset, SCREEN_HEIGHT / 2 - 32); // mouth line-3
	ghrna.drawLine(SCREEN_WIDTH / 2 - 30 + offset, SCREEN_HEIGHT / 2 - 43, SCREEN_WIDTH / 2 - 15 + offset, SCREEN_HEIGHT / 2 - 33); // mouth line-3

	ghrna.drawLine(SCREEN_WIDTH / 2 - 15 + offset, SCREEN_HEIGHT / 2 - 30, SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 40); // mouth line-4
	ghrna.drawLine(SCREEN_WIDTH / 2 - 15 + offset, SCREEN_HEIGHT / 2 - 31, SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 41); // mouth line-4
	ghrna.drawLine(SCREEN_WIDTH / 2 - 15 + offset, SCREEN_HEIGHT / 2 - 32, SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 42); // mouth line-4
	ghrna.drawLine(SCREEN_WIDTH / 2 - 15 + offset, SCREEN_HEIGHT / 2 - 33, SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 43); // mouth line-4

	ghrna.drawLine(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 40, SCREEN_WIDTH / 2 + 15 + offset, SCREEN_HEIGHT / 2 - 30); // mouth line-5
	ghrna.drawLine(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 41, SCREEN_WIDTH / 2 + 15 + offset, SCREEN_HEIGHT / 2 - 31); // mouth line-5
	ghrna.drawLine(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 42, SCREEN_WIDTH / 2 + 15 + offset, SCREEN_HEIGHT / 2 - 32); // mouth line-5
	ghrna.drawLine(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 43, SCREEN_WIDTH / 2 + 15 + offset, SCREEN_HEIGHT / 2 - 33); // mouth line-5

	ghrna.drawLine(SCREEN_WIDTH / 2 + 15 + offset, SCREEN_HEIGHT / 2 - 30, SCREEN_WIDTH / 2 + 30 + offset, SCREEN_HEIGHT / 2 - 40); // mouth line-6
	ghrna.drawLine(SCREEN_WIDTH / 2 + 15 + offset, SCREEN_HEIGHT / 2 - 31, SCREEN_WIDTH / 2 + 30 + offset, SCREEN_HEIGHT / 2 - 41); // mouth line-6
	ghrna.drawLine(SCREEN_WIDTH / 2 + 15 + offset, SCREEN_HEIGHT / 2 - 32, SCREEN_WIDTH / 2 + 30 + offset, SCREEN_HEIGHT / 2 - 42); // mouth line-6
	ghrna.drawLine(SCREEN_WIDTH / 2 + 15 + offset, SCREEN_HEIGHT / 2 - 33, SCREEN_WIDTH / 2 + 30 + offset, SCREEN_HEIGHT / 2 - 43); // mouth line-6

	ghrna.drawLine(SCREEN_WIDTH / 2 + 30 + offset, SCREEN_HEIGHT / 2 - 40, SCREEN_WIDTH / 2 + 45 + offset, SCREEN_HEIGHT / 2 - 30); // mouth line-7
	ghrna.drawLine(SCREEN_WIDTH / 2 + 30 + offset, SCREEN_HEIGHT / 2 - 41, SCREEN_WIDTH / 2 + 45 + offset, SCREEN_HEIGHT / 2 - 31); // mouth line-7
	ghrna.drawLine(SCREEN_WIDTH / 2 + 30 + offset, SCREEN_HEIGHT / 2 - 42, SCREEN_WIDTH / 2 + 45 + offset, SCREEN_HEIGHT / 2 - 32); // mouth line-7
	ghrna.drawLine(SCREEN_WIDTH / 2 + 30 + offset, SCREEN_HEIGHT / 2 - 43, SCREEN_WIDTH / 2 + 45 + offset, SCREEN_HEIGHT / 2 - 33); // mouth line-7

	ghrna.drawLine(SCREEN_WIDTH / 2 + 45 + offset, SCREEN_HEIGHT / 2 - 30, SCREEN_WIDTH / 2 + 60 + offset, SCREEN_HEIGHT / 2 - 40); // mouth line-8
	ghrna.drawLine(SCREEN_WIDTH / 2 + 45 + offset, SCREEN_HEIGHT / 2 - 31, SCREEN_WIDTH / 2 + 60 + offset, SCREEN_HEIGHT / 2 - 41); // mouth line-8
	ghrna.drawLine(SCREEN_WIDTH / 2 + 45 + offset, SCREEN_HEIGHT / 2 - 32, SCREEN_WIDTH / 2 + 60 + offset, SCREEN_HEIGHT / 2 - 42); // mouth line-8
	ghrna.drawLine(SCREEN_WIDTH / 2 + 45 + offset, SCREEN_HEIGHT / 2 - 33, SCREEN_WIDTH / 2 + 60 + offset, SCREEN_HEIGHT / 2 - 43); // mouth line-8
}

Disgust::Disgust()
{
}


Disgust::~Disgust()
{
}
