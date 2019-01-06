#include "Neutral.h"

void Neutral::neutralFace_1()
{
	Draw tatasth; 
	glColor3f(0.897f, 0.7895f, 0.12344f);
	tatasth.drawCircle(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, 0, 120, 100, 0.867, 1); // face
	glColor3d(0, 0, 0);
	tatasth.drawCircle(SCREEN_WIDTH / 2 + 48, SCREEN_HEIGHT / 2 + 50, 0, 20, 100, 0.7, 1); // right eye
	tatasth.drawCircle(SCREEN_WIDTH / 2 - 50, SCREEN_HEIGHT / 2 + 50, 0, 20, 100, 0.7, 1); // left eye
	glColor3d(0, 0, 0);
	glColor3f(0.0f, 0.0f, 0.0f);

	tatasth.drawLine(SCREEN_WIDTH / 2 - 35, SCREEN_HEIGHT / 2 - 40, SCREEN_WIDTH / 2 + 40, SCREEN_HEIGHT / 2 - 40); // mouth 
	tatasth.drawLine(SCREEN_WIDTH / 2 - 35, SCREEN_HEIGHT / 2 - 41, SCREEN_WIDTH / 2 + 40, SCREEN_HEIGHT / 2 - 41); // mouth 
	tatasth.drawLine(SCREEN_WIDTH / 2 - 35, SCREEN_HEIGHT / 2 - 42, SCREEN_WIDTH / 2 + 40, SCREEN_HEIGHT / 2 - 42); // mouth 
	tatasth.drawLine(SCREEN_WIDTH / 2 - 35, SCREEN_HEIGHT / 2 - 43, SCREEN_WIDTH / 2 + 40, SCREEN_HEIGHT / 2 - 43); // mouth 
	tatasth.drawLine(SCREEN_WIDTH / 2 - 35, SCREEN_HEIGHT / 2 - 44, SCREEN_WIDTH / 2 + 40, SCREEN_HEIGHT / 2 - 44); // mouth 
	//tatasth.drawLine(SCREEN_WIDTH / 2 - 35, SCREEN_HEIGHT / 2 - 45, SCREEN_WIDTH / 2 + 40, SCREEN_HEIGHT / 2 - 45); // mouth
}


void Neutral::neutralFace_3(int offset)
{
	Draw tatasth;

	glColor3f(0.5f, 0.5f, 0.5f);//grey
	tatasth.drawCircle(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2, 0, 130, 90, 1, 1);  //  face
	glColor3d(1, 1, 1);//white
	tatasth.drawCircle(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2, 0, 120, 90, 1, 1);  //  face
	glColor3d(0, 0, 0);//black
	tatasth.drawCircle(SCREEN_WIDTH / 2 - 40 + offset, SCREEN_HEIGHT / 2 + 30, 0, 20, 90, 1, 1); // left outer eye
	tatasth.drawCircle(SCREEN_WIDTH / 2 + 40 + offset, SCREEN_HEIGHT / 2 + 30, 0, 20, 90, 1, 1); // right outer eye

	glColor3d(1, 1, 1);//white
	tatasth.drawCircle(SCREEN_WIDTH / 2 - 40 + offset, SCREEN_HEIGHT / 2 + 30, 0, 15, 90, 1, 1); // left inner eye
	tatasth.drawCircle(SCREEN_WIDTH / 2 + 40 + offset, SCREEN_HEIGHT / 2 + 30, 0, 15, 90, 1, 1); // right inner eye
	glColor3f(0.0f, 0.0f, 0.0f);
    
	tatasth.drawLine(SCREEN_WIDTH / 2 - 49 + offset, SCREEN_HEIGHT / 2 + 31, SCREEN_WIDTH / 2 - 31 + offset, SCREEN_HEIGHT / 2 + 31); // left eye
	tatasth.drawLine(SCREEN_WIDTH / 2 - 50 + offset, SCREEN_HEIGHT / 2 + 30, SCREEN_WIDTH / 2 - 30 + offset, SCREEN_HEIGHT / 2 + 30); // left eye
	tatasth.drawLine(SCREEN_WIDTH / 2 - 49 + offset, SCREEN_HEIGHT / 2 + 29, SCREEN_WIDTH / 2 - 31 + offset, SCREEN_HEIGHT / 2 + 29); // left eye

	tatasth.drawLine(SCREEN_WIDTH / 2 + 49 + offset, SCREEN_HEIGHT / 2 + 31, SCREEN_WIDTH / 2 + 31 + offset, SCREEN_HEIGHT / 2 + 31); // right eye
	tatasth.drawLine(SCREEN_WIDTH / 2 + 50 + offset, SCREEN_HEIGHT / 2 + 30, SCREEN_WIDTH / 2 + 30 + offset, SCREEN_HEIGHT / 2 + 30); // right eye
	tatasth.drawLine(SCREEN_WIDTH / 2 + 49 + offset, SCREEN_HEIGHT / 2 + 29, SCREEN_WIDTH / 2 + 31 + offset, SCREEN_HEIGHT / 2 + 29); // right eye

	tatasth.drawLine(SCREEN_WIDTH / 2 - 33 + offset, SCREEN_HEIGHT / 2 - 40, SCREEN_WIDTH / 2 + 38 + offset, SCREEN_HEIGHT / 2 - 40); // mouth 
	tatasth.drawLine(SCREEN_WIDTH / 2 - 34 + offset, SCREEN_HEIGHT / 2 - 41, SCREEN_WIDTH / 2 + 39 + offset, SCREEN_HEIGHT / 2 - 41); // mouth 
	tatasth.drawLine(SCREEN_WIDTH / 2 - 35 + offset, SCREEN_HEIGHT / 2 - 42, SCREEN_WIDTH / 2 + 40 + offset, SCREEN_HEIGHT / 2 - 42); // mouth 
	tatasth.drawLine(SCREEN_WIDTH / 2 - 34 + offset, SCREEN_HEIGHT / 2 - 43, SCREEN_WIDTH / 2 + 39 + offset, SCREEN_HEIGHT / 2 - 43); // mouth 
	tatasth.drawLine(SCREEN_WIDTH / 2 - 33 + offset, SCREEN_HEIGHT / 2 - 44, SCREEN_WIDTH / 2 + 38 + offset, SCREEN_HEIGHT / 2 - 44); // mouth 
}

void Neutral::neutralFace_2(int offset)
{
	Draw tatasth;
	glColor3f(0.897f, 0.7895f, 0.12344f);
	//glColor3f(1.0f, 0.2588244f, 0.0f);
	tatasth.drawCircle(SCREEN_WIDTH / 2 + offset, SCREEN_HEIGHT / 2, 0, 120, 100, 0.867, 1); // face
	glColor3d(0, 0, 0);
	glColor3d(0, 0, 0);
	glColor3f(0.0f, 0.0f, 0.0f);


	tatasth.drawLine(SCREEN_WIDTH / 2 - 60 + offset, SCREEN_HEIGHT / 2 + 50, SCREEN_WIDTH / 2 - 20 + offset, SCREEN_HEIGHT / 2 + 50); // left eye 
	tatasth.drawLine(SCREEN_WIDTH / 2 - 60 + offset, SCREEN_HEIGHT / 2 + 51, SCREEN_WIDTH / 2 - 20 + offset, SCREEN_HEIGHT / 2 + 51); // left eye 
	tatasth.drawLine(SCREEN_WIDTH / 2 - 60 + offset, SCREEN_HEIGHT / 2 + 52, SCREEN_WIDTH / 2 - 20 + offset, SCREEN_HEIGHT / 2 + 52); // left eye 
	tatasth.drawLine(SCREEN_WIDTH / 2 - 60 + offset, SCREEN_HEIGHT / 2 + 53, SCREEN_WIDTH / 2 - 20 + offset, SCREEN_HEIGHT / 2 + 53); // left eye
	tatasth.drawLine(SCREEN_WIDTH / 2 - 60 + offset, SCREEN_HEIGHT / 2 + 54, SCREEN_WIDTH / 2 - 20 + offset, SCREEN_HEIGHT / 2 + 54); // left eye 
	//tatasth.drawLine(SCREEN_WIDTH / 2 - 60 + offset, SCREEN_HEIGHT / 2 + 55, SCREEN_WIDTH / 2 - 20 + offset, SCREEN_HEIGHT / 2 + 55); // left eye


	tatasth.drawLine(SCREEN_WIDTH / 2 + 20 + offset, SCREEN_HEIGHT / 2 + 50, SCREEN_WIDTH / 2 + 60 + offset, SCREEN_HEIGHT / 2 + 50); //right eye 
	tatasth.drawLine(SCREEN_WIDTH / 2 + 20 + offset, SCREEN_HEIGHT / 2 + 51, SCREEN_WIDTH / 2 + 60 + offset, SCREEN_HEIGHT / 2 + 51); //right eye 
	tatasth.drawLine(SCREEN_WIDTH / 2 + 20 + offset, SCREEN_HEIGHT / 2 + 52, SCREEN_WIDTH / 2 + 60 + offset, SCREEN_HEIGHT / 2 + 52); //right eye 
	tatasth.drawLine(SCREEN_WIDTH / 2 + 20 + offset, SCREEN_HEIGHT / 2 + 53, SCREEN_WIDTH / 2 + 60 + offset, SCREEN_HEIGHT / 2 + 53); //right eye
	tatasth.drawLine(SCREEN_WIDTH / 2 + 20 + offset, SCREEN_HEIGHT / 2 + 54, SCREEN_WIDTH / 2 + 60 + offset, SCREEN_HEIGHT / 2 + 54); //right eye
	//tatasth.drawLine(SCREEN_WIDTH / 2 + 20 + offset, SCREEN_HEIGHT / 2 + 55, SCREEN_WIDTH / 2 + 60 + offset, SCREEN_HEIGHT / 2 + 55); //right eye


	tatasth.drawLine(SCREEN_WIDTH / 2 - 35 + offset, SCREEN_HEIGHT / 2 - 40, SCREEN_WIDTH / 2 + 40 + offset, SCREEN_HEIGHT / 2 - 40); // mouth 
	tatasth.drawLine(SCREEN_WIDTH / 2 - 35 + offset, SCREEN_HEIGHT / 2 - 41, SCREEN_WIDTH / 2 + 40 + offset, SCREEN_HEIGHT / 2 - 41); // mouth 
	tatasth.drawLine(SCREEN_WIDTH / 2 - 35 + offset, SCREEN_HEIGHT / 2 - 42, SCREEN_WIDTH / 2 + 40 + offset, SCREEN_HEIGHT / 2 - 42); // mouth 
	tatasth.drawLine(SCREEN_WIDTH / 2 - 35 + offset, SCREEN_HEIGHT / 2 - 43, SCREEN_WIDTH / 2 + 40 + offset, SCREEN_HEIGHT / 2 - 43); // mouth 
	tatasth.drawLine(SCREEN_WIDTH / 2 - 35 + offset, SCREEN_HEIGHT / 2 - 44, SCREEN_WIDTH / 2 + 40 + offset, SCREEN_HEIGHT / 2 - 44); // mouth 
	//tatasth.drawLine(SCREEN_WIDTH / 2 - 35 + offset, SCREEN_HEIGHT / 2 - 45, SCREEN_WIDTH / 2 + 40 + offset, SCREEN_HEIGHT / 2 - 45); // mouth
}

Neutral::Neutral()
{
}


Neutral::~Neutral()
{
}
