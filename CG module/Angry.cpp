#include "Angry.h"

Angry::Angry()
{
}

void Angry::angryFace_2(int offset)
{
	Draw gussa;

	glColor3f(1.0f, 0.2588244f, 0.0f);
	gussa.drawCircle(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2, 0, 120, 90, 1, 1);  //  face
	glColor3d(0, 0, 0);
	gussa.drawCircle(SCREEN_WIDTH / 2 + 48 + offset, SCREEN_HEIGHT / 2, 0, 10, 100, 0.7, 1); // right
	gussa.drawCircle(SCREEN_WIDTH / 2 - 50 + offset, SCREEN_HEIGHT / 2, 0, 10, 100, 0.7, 1); // left
	glColor3f(0.0f, 0.0f, 0.0f);

	gussa.drawArc(SCREEN_WIDTH / 2 - 50 + offset, SCREEN_HEIGHT / 2 + 30, 0, 21, 190, 310, 50); // left eyebrow
	//gussa.drawLine()
	gussa.drawArc(SCREEN_WIDTH / 2 - 50 + offset, SCREEN_HEIGHT / 2 + 30, 0, 22, 190, 310, 50); // left eyebrow
	gussa.drawArc(SCREEN_WIDTH / 2 - 50 + offset, SCREEN_HEIGHT / 2 + 30, 0, 23, 190, 310, 50); // left eyebrow
	gussa.drawArc(SCREEN_WIDTH / 2 - 50 + offset, SCREEN_HEIGHT / 2 + 30, 0, 24, 190, 310, 50); // left eyebrow
	gussa.drawArc(SCREEN_WIDTH / 2 - 50 + offset, SCREEN_HEIGHT / 2 + 30, 0, 25, 190, 310, 50); // left eyebrow

	gussa.drawArc(SCREEN_WIDTH / 2 + 48 + offset, SCREEN_HEIGHT / 2 + 30, 0, 21, 230, 350, 50); // right eyebrow
	gussa.drawArc(SCREEN_WIDTH / 2 + 48 + offset, SCREEN_HEIGHT / 2 + 30, 0, 22, 230, 350, 50); // right eyebrow
	gussa.drawArc(SCREEN_WIDTH / 2 + 48 + offset, SCREEN_HEIGHT / 2 + 30, 0, 23, 230, 350, 50); // right eyebrow
	gussa.drawArc(SCREEN_WIDTH / 2 + 48 + offset, SCREEN_HEIGHT / 2 + 30, 0, 24, 230, 350, 50); // right eyebrow
	gussa.drawArc(SCREEN_WIDTH / 2 + 48 + offset, SCREEN_HEIGHT / 2 + 30, 0, 25, 230, 350, 50); // right eyebrow

	gussa.drawLine(SCREEN_WIDTH / 2 - 25 + offset, SCREEN_HEIGHT / 2 - 40, SCREEN_WIDTH / 2 + 30 + offset, SCREEN_HEIGHT / 2 - 40); // mouth 
	gussa.drawLine(SCREEN_WIDTH / 2 - 25 + offset, SCREEN_HEIGHT / 2 - 41, SCREEN_WIDTH / 2 + 30 + offset, SCREEN_HEIGHT / 2 - 41); // mouth 
	gussa.drawLine(SCREEN_WIDTH / 2 - 25 + offset, SCREEN_HEIGHT / 2 - 42, SCREEN_WIDTH / 2 + 30 + offset, SCREEN_HEIGHT / 2 - 42); // mouth 
	gussa.drawLine(SCREEN_WIDTH / 2 - 25 + offset, SCREEN_HEIGHT / 2 - 43, SCREEN_WIDTH / 2 + 30 + offset, SCREEN_HEIGHT / 2 - 43); // mouth 
	gussa.drawLine(SCREEN_WIDTH / 2 - 25 + offset, SCREEN_HEIGHT / 2 - 44, SCREEN_WIDTH / 2 + 30 + offset, SCREEN_HEIGHT / 2 - 44); // mouth
}


void Angry::angryFace_3(int offset)
{
	Draw gussa;

	glColor3f(1.0f, 0.2588244f, 0.0f);
	//glColor3f(0.897f, 0.7895f, 0.12344f);
	gussa.drawCircle(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2, 0, 120, 90, 1, 1);  //  face
	glColor3d(0, 0, 0);
	gussa.drawCircle(SCREEN_WIDTH / 2 + 38 + offset, SCREEN_HEIGHT / 2 + 20, 0, 20, 100, 0.7, 1); // right
	gussa.drawCircle(SCREEN_WIDTH / 2 - 40 + offset, SCREEN_HEIGHT / 2 + 20, 0, 20, 100, 0.7, 1); // left
	glColor3f(0.0f, 0.0f, 0.0f);

	gussa.drawLine(SCREEN_WIDTH / 2 + 18 + offset, SCREEN_HEIGHT / 2 + 25, SCREEN_WIDTH / 2 + 68 + offset, SCREEN_HEIGHT / 2 + 45); // right eyebrow-1
	gussa.drawLine(SCREEN_WIDTH / 2 + 18 + offset, SCREEN_HEIGHT / 2 + 26, SCREEN_WIDTH / 2 + 68 + offset, SCREEN_HEIGHT / 2 + 46); // right eyebrow-1
	gussa.drawLine(SCREEN_WIDTH / 2 + 18 + offset, SCREEN_HEIGHT / 2 + 27, SCREEN_WIDTH / 2 + 68 + offset, SCREEN_HEIGHT / 2 + 47); // right eyebrow-1
	gussa.drawLine(SCREEN_WIDTH / 2 + 18 + offset, SCREEN_HEIGHT / 2 + 28, SCREEN_WIDTH / 2 + 68 + offset, SCREEN_HEIGHT / 2 + 48); // right eyebrow-1
	gussa.drawLine(SCREEN_WIDTH / 2 + 18 + offset, SCREEN_HEIGHT / 2 + 29, SCREEN_WIDTH / 2 + 68 + offset, SCREEN_HEIGHT / 2 + 49); // right eyebrow-1
	gussa.drawLine(SCREEN_WIDTH / 2 + 18 + offset, SCREEN_HEIGHT / 2 + 30, SCREEN_WIDTH / 2 + 68 + offset, SCREEN_HEIGHT / 2 + 50); // right eyebrow-1
	gussa.drawLine(SCREEN_WIDTH / 2 + 18 + offset, SCREEN_HEIGHT / 2 + 25, SCREEN_WIDTH / 2 + 18 + offset, SCREEN_HEIGHT / 2 + 43); // right eyebrow-2
	gussa.drawLine(SCREEN_WIDTH / 2 + 17 + offset, SCREEN_HEIGHT / 2 + 26, SCREEN_WIDTH / 2 + 17 + offset, SCREEN_HEIGHT / 2 + 42); // right eyebrow-2
	gussa.drawLine(SCREEN_WIDTH / 2 + 16 + offset, SCREEN_HEIGHT / 2 + 27, SCREEN_WIDTH / 2 + 16 + offset, SCREEN_HEIGHT / 2 + 41); // right eyebrow-2

	gussa.drawLine(SCREEN_WIDTH / 2 - 18 + offset, SCREEN_HEIGHT / 2 + 25, SCREEN_WIDTH / 2 - 68 + offset, SCREEN_HEIGHT / 2 + 45); // left eyebrow-1
	gussa.drawLine(SCREEN_WIDTH / 2 - 18 + offset, SCREEN_HEIGHT / 2 + 26, SCREEN_WIDTH / 2 - 68 + offset, SCREEN_HEIGHT / 2 + 46); // left eyebrow-1
	gussa.drawLine(SCREEN_WIDTH / 2 - 18 + offset, SCREEN_HEIGHT / 2 + 27, SCREEN_WIDTH / 2 - 68 + offset, SCREEN_HEIGHT / 2 + 47); // left eyebrow-1
	gussa.drawLine(SCREEN_WIDTH / 2 - 18 + offset, SCREEN_HEIGHT / 2 + 28, SCREEN_WIDTH / 2 - 68 + offset, SCREEN_HEIGHT / 2 + 48); // left eyebrow-1
	gussa.drawLine(SCREEN_WIDTH / 2 - 18 + offset, SCREEN_HEIGHT / 2 + 29, SCREEN_WIDTH / 2 - 68 + offset, SCREEN_HEIGHT / 2 + 49); // left eyebrow-1
	gussa.drawLine(SCREEN_WIDTH / 2 - 18 + offset, SCREEN_HEIGHT / 2 + 30, SCREEN_WIDTH / 2 - 68 + offset, SCREEN_HEIGHT / 2 + 50); // left eyebrow-1
	gussa.drawLine(SCREEN_WIDTH / 2 - 18 + offset, SCREEN_HEIGHT / 2 + 25, SCREEN_WIDTH / 2 - 18 + offset, SCREEN_HEIGHT / 2 + 43); // left eyebrow-2
	gussa.drawLine(SCREEN_WIDTH / 2 - 17 + offset, SCREEN_HEIGHT / 2 + 26, SCREEN_WIDTH / 2 - 17 + offset, SCREEN_HEIGHT / 2 + 42); // left eyebrow-2
	gussa.drawLine(SCREEN_WIDTH / 2 - 16 + offset, SCREEN_HEIGHT / 2 + 27, SCREEN_WIDTH / 2 - 16 + offset, SCREEN_HEIGHT / 2 + 41); // left eyebrow-2

	GLfloat r1 = 55, theta1 = 40, theta2 = 140, xc, yc;
	xc = SCREEN_WIDTH / 2.0f;
	yc = SCREEN_HEIGHT / 2.0f - 70.0f;
	gussa.drawArc(xc + offset, yc, 0, r1, theta1, theta2, 50); //mouth
	gussa.drawArc(xc + offset, yc, 0, r1 - 1, theta1, theta2, 50); //mouth
	gussa.drawArc(xc + offset, yc, 0, r1 - 2, theta1, theta2, 50); //mouth
	gussa.drawArc(xc + offset, yc, 0, r1 - 3, theta1, theta2, 50); //mouth
	gussa.drawArc(xc + offset, yc, 0, r1 - 4, theta1, theta2, 50); //mouth
}

void Angry::angryFace_1()
{
	Draw gussa;


	//glColor3f(0.897f, 0.7895f, 0.12344f);
	glColor3f(1.0f, 0.2588244f, 0.0f);
	gussa.drawCircle(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, 0, 120, 90, 0.867, 1);  //  face
	glColor3d(1, 1, 1);
	gussa.drawCircle(SCREEN_WIDTH / 2 + 38, SCREEN_HEIGHT / 2 + 50, 0, 20, 100, 0.7, 1); // right
	gussa.drawCircle(SCREEN_WIDTH / 2 - 40, SCREEN_HEIGHT / 2 + 50, 0, 20, 100, 0.7, 1); // left
	glColor3f(0.0f, 0.0f, 0.0f);
	gussa.drawArc(SCREEN_WIDTH / 2 - 45, SCREEN_HEIGHT / 2 + 50, 0, 30, 40, 100, 50); // left eyebrow
	gussa.drawArc(SCREEN_WIDTH / 2 - 45, SCREEN_HEIGHT / 2 + 50, 0, 31, 40, 100, 50); // left eyebrow
	gussa.drawArc(SCREEN_WIDTH / 2 - 45, SCREEN_HEIGHT / 2 + 50, 0, 32, 40, 100, 50); // left eyebrow
	gussa.drawArc(SCREEN_WIDTH / 2 - 45, SCREEN_HEIGHT / 2 + 50, 0, 33, 40, 100, 50); // left eyebrow
	gussa.drawArc(SCREEN_WIDTH / 2 + 45, SCREEN_HEIGHT / 2 + 50, 0, 30, 90, 150, 50); // right eyebrow
	gussa.drawArc(SCREEN_WIDTH / 2 + 45, SCREEN_HEIGHT / 2 + 50, 0, 31, 90, 150, 50); // right eyebrow
	gussa.drawArc(SCREEN_WIDTH / 2 + 45, SCREEN_HEIGHT / 2 + 50, 0, 32, 90, 150, 50); // right eyebrow
	gussa.drawArc(SCREEN_WIDTH / 2 + 45, SCREEN_HEIGHT / 2 + 50, 0, 33, 90, 150, 50); // right eyebrow

	GLfloat r1 = 45, theta1 = 30, theta2 = 150, xc, yc;
	xc = SCREEN_WIDTH / 2.0f;
	yc = SCREEN_HEIGHT / 2.0f - 70.0f;

	gussa.drawArc(xc, yc, 0, r1, theta1, theta2, 50); //mouth


	//code for second arc 

	/*GLfloat r1 = 45, theta1 = 30, theta2 = 150, x1, x2, y1, y2, alpha1, alpha2, r2 = 40, x, y, x3, y3, phi, xc, yc;

	x1 = r1 * cos((M_PI / 180)*theta1) + 320;
	x2 = r1 * cos((M_PI / 180)*theta2) + 320;
	y1 = r1 * sin((M_PI / 180)*theta1) + 170;
	y2 = r1 * sin((M_PI / 180)*theta2) + 170;


	x3 = (x1 + x2) / 2;
	y3 = (y1 + y2) / 2;
	GLfloat q = sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));

	x = x3 - sqrt(abs(pow(r2, 2) - pow(q / 2, 2)))*((y1 - y2) / q);
	y = y3 - sqrt(abs(pow(r2, 2) - pow(q / 2, 2)))*((x2 - x1) / q);


	alpha1 = (180 / M_PI) * asin((y2 - y) / r2); // start angle
	phi = asin(((y1 - y) / r2));
	alpha2 = 180 - ((180 / M_PI) * phi);//end angle 


	//drawSecondArc(r1, theta1, theta2, xc, yc, r2);
	//gussa.drawArc(x, y, 0, r2, alpha1, alpha2, 50); //mouth

	*/

}

Angry::~Angry()
{
}
