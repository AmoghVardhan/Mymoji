#include "Surprise.h"


void Surprise::SurpriseFace_1()
{
	Draw achambhit;

	glColor3f(0.897f, 0.7895f, 0.12344f);
	achambhit.drawCircle(SCREEN_WIDTH / 2 + 215, SCREEN_HEIGHT / 2, 0, 120, 100, 0.867, 1);
	glColor3d(0, 0, 0);
	achambhit.drawCircle(SCREEN_WIDTH / 2 + 48 + 215, SCREEN_HEIGHT / 2 + 50, 0, 20, 100, 0.7, 1); //RIGHT EYE
	achambhit.drawCircle(SCREEN_WIDTH / 2 - 50 + 215, SCREEN_HEIGHT / 2 + 50, 0, 20, 100, 0.7, 1); //LEFT EYE
	glColor3f(0.0f, 0.0f, 0.0f);

	//drawCircleHollow( SCREEN_WIDTH / 2, SCREEN_HEIGHT /2-50 , 0, 20, 36 );
	achambhit.drawArc(SCREEN_WIDTH / 2 - 20 + 280 + 20 + 10, SCREEN_HEIGHT / 2 - 130, 0, 90, 100, 150, 100); //mouth
	achambhit.drawArc(SCREEN_WIDTH / 2 - 20 + 280 + 20 + 10, SCREEN_HEIGHT / 2 - 130, 0, 91, 100, 150, 100); //mouth
	achambhit.drawArc(SCREEN_WIDTH / 2 - 20 + 280 + 20 + 10, SCREEN_HEIGHT / 2 - 130, 0, 92, 100, 150, 100); //mouth
	achambhit.drawArc(SCREEN_WIDTH / 2 - 20 + 280 + 20 + 10, SCREEN_HEIGHT / 2 - 130, 0, 93, 100, 150, 100); //mouth
}

void Surprise::SurpriseFace_2(int offset)
{
	Draw achambhit;

	glColor3f(0.897f, 0.7895f, 0.12344f);
	//glColor3f(1.0f, 0.2588244f, 0.0f);
	achambhit.drawCircle(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2, 0, 120, 90, 1, 1);  //  face
	glColor3d(1, 1, 1);//white
	achambhit.drawCircle(SCREEN_WIDTH / 2 + 38 + offset, SCREEN_HEIGHT / 2 + 40, 0, 30, 100, 1, 1); // right
	achambhit.drawCircle(SCREEN_WIDTH / 2 - 40 + offset, SCREEN_HEIGHT / 2 + 40, 0, 30, 100, 1, 1); // left
	glColor3d(0, 0, 0);
	achambhit.drawCircle(SCREEN_WIDTH / 2 + 38 + offset, SCREEN_HEIGHT / 2 + 40, 0, 10, 100, 1, 1); // right
	achambhit.drawCircle(SCREEN_WIDTH / 2 - 40 + offset, SCREEN_HEIGHT / 2 + 40, 0, 10, 100, 1, 1); // left
	glColor3f(0.0f, 0.0f, 0.0f);

	achambhit.drawCircle(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2 - 40, 0, 20, 100, 1.5, 1); // mouth

	//glColor3f(0.85f, 0.85f, 0.10f);
	achambhit.drawArc(SCREEN_WIDTH / 2 - 33 + offset, SCREEN_HEIGHT / 2 + 60, 0, 35, 70, 150, 50); // left eyebrow
	achambhit.drawArc(SCREEN_WIDTH / 2 - 33 + offset, SCREEN_HEIGHT / 2 + 60, 0, 36, 70, 150, 50); // left eyebrow
	achambhit.drawArc(SCREEN_WIDTH / 2 - 33 + offset, SCREEN_HEIGHT / 2 + 60, 0, 37, 70, 150, 50); // left eyebrow
	achambhit.drawArc(SCREEN_WIDTH / 2 - 33 + offset, SCREEN_HEIGHT / 2 + 60, 0, 38, 70, 150, 50); // left eyebrow

	achambhit.drawArc(SCREEN_WIDTH / 2 + 33 + offset, SCREEN_HEIGHT / 2 + 60, 0, 35, 30, 110, 50); // right eyebrow
	achambhit.drawArc(SCREEN_WIDTH / 2 + 33 + offset, SCREEN_HEIGHT / 2 + 60, 0, 36, 30, 110, 50); // right eyebrow
	achambhit.drawArc(SCREEN_WIDTH / 2 + 33 + offset, SCREEN_HEIGHT / 2 + 60, 0, 37, 30, 110, 50); // right eyebrow
	achambhit.drawArc(SCREEN_WIDTH / 2 + 33 + offset, SCREEN_HEIGHT / 2 + 60, 0, 38, 30, 110, 50); // right eyebrow
}

Surprise::Surprise()
{
}


Surprise::~Surprise()
{
}
