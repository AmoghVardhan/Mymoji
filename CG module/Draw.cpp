#include "Draw.h"

Draw::Draw()
{

}

void Draw::drawArc(GLfloat x, GLfloat y, GLfloat z, GLfloat radius, int start, int end, GLint numberOfSides)
{
	GLint numberOfVertices = numberOfSides + 1;
	GLfloat doublePi = M_PI * (-start + end) / 180.0f;


	GLfloat start_rad = M_PI * start / 180.0f;
	GLfloat circleVerticesX[1000];
	GLfloat circleVerticesY[1000];
	GLfloat circleVerticesZ[1000];
	GLfloat allCircleVertices[1000];


	//circleVerticesX[0] = x;
	//circleVerticesY[0] = y;
	//circleVerticesZ[0] = z;

	for (int i = 0; i < numberOfVertices; i++)
	{
		circleVerticesX[i] = x + (radius * cos(start_rad + i * doublePi / numberOfSides));
		circleVerticesY[i] = y + (radius * sin(start_rad + i * doublePi / numberOfSides));
		circleVerticesZ[i] = z;
	}

	/*int index=0;
	 for ( int i = start; i <= end; i++ )
	 {
	 circleVerticesX[index] = x + ( radius * cos(i *  M_PI/180.0));
	 circleVerticesY[index] = y + ( radius * sin (i *  M_PI/180.0));
	 printf("...%f...\n",circleVerticesY[index]);
	 circleVerticesZ[index++] = z;
	 }*/

	 //GLfloat allCircleVertices[numberOfVertices * 3];
	//GLfloat *allCircleVertices = new GLfloat[numberOfVertices * 3];

	for (int i = 0; i < numberOfVertices; i++)
	{
		allCircleVertices[i * 3] = circleVerticesX[i];
		allCircleVertices[(i * 3) + 1] = circleVerticesY[i];
		allCircleVertices[(i * 3) + 2] = circleVerticesZ[i];
	}

	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, allCircleVertices);
	glDrawArrays(GL_LINE_STRIP, 0, numberOfVertices);
	glDisableClientState(GL_VERTEX_ARRAY);

	
}



void Draw::drawCircle(GLfloat x, GLfloat y, GLfloat z, GLfloat radius, GLint numberOfSides, GLfloat factorx, GLfloat factory)
{
	int numberOfVertices = numberOfSides + 2;

	GLfloat twicePi = 2.0f * M_PI;

	GLfloat circleVerticesX[1000];
	GLfloat circleVerticesY[1000];
	GLfloat circleVerticesZ[1000];
	GLfloat allCircleVertices[1000];


	circleVerticesX[0] = x;
	circleVerticesY[0] = y;
	circleVerticesZ[0] = z;

	for (int i = 1; i < numberOfVertices; i++)
	{
		circleVerticesX[i] = x + (radius * cos(i *  twicePi / numberOfSides))* factorx;
		circleVerticesY[i] = y + (radius * sin(i * twicePi / numberOfSides))* factory;
		circleVerticesZ[i] = z;
	}


	for (int i = 0; i < numberOfVertices; i++)
	{
		allCircleVertices[i * 3] = circleVerticesX[i];
		allCircleVertices[(i * 3) + 1] = circleVerticesY[i];
		allCircleVertices[(i * 3) + 2] = circleVerticesZ[i];
	}

	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, allCircleVertices);
	glDrawArrays(GL_TRIANGLE_FAN, 0, numberOfVertices);
	glDisableClientState(GL_VERTEX_ARRAY);


}


void Draw::drawSemiCircle(GLfloat x, GLfloat y, GLfloat z, GLfloat radius, GLint numberOfSides, GLfloat factorx, GLfloat factory)
{
	int numberOfVertices = numberOfSides + 2;

	GLfloat twicePi = 1.0f * M_PI;

	GLfloat circleVerticesX[1000];
	GLfloat circleVerticesY[1000];
	GLfloat circleVerticesZ[1000];
	GLfloat allCircleVertices[1000];


	circleVerticesX[0] = x;
	circleVerticesY[0] = y;
	circleVerticesZ[0] = z;

	for (int i = 1; i < numberOfVertices; i++)
	{
		circleVerticesX[i] = x + (radius * cos(i *  twicePi / numberOfSides))* factorx;
		circleVerticesY[i] = y + (radius * sin(i * twicePi / numberOfSides))* factory;
		/*if (cos(i * twicePi / numberOfSides) < 0)
			circleVerticesY[i] = y + 3 + (radius * sin(i * twicePi / numberOfSides))* factory;
		else if (cos(i * twicePi / numberOfSides) == 0)
			circleVerticesY[i] = y + 15 + (radius * sin(i * twicePi / numberOfSides))* factory;
		else
		    circleVerticesY[i] = y - 3 + (radius * sin(i * twicePi / numberOfSides))* factory;*/
		circleVerticesZ[i] = z;
	}


	for (int i = 0; i < numberOfVertices; i++)
	{
		allCircleVertices[i * 3] = circleVerticesX[i];
		allCircleVertices[(i * 3) + 1] = circleVerticesY[i];
		allCircleVertices[(i * 3) + 2] = circleVerticesZ[i];
	}

	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, allCircleVertices);
	glDrawArrays(GL_TRIANGLE_FAN, 0, numberOfVertices);
	glDisableClientState(GL_VERTEX_ARRAY);


}


void Draw::drawCircleHollow(GLfloat x, GLfloat y, GLfloat z, GLfloat radius, GLint numberOfSides)
{
	GLint numberOfVertices = numberOfSides + 1;

	GLfloat doublePi = 2.0f * M_PI;

	GLfloat circleVerticesX[1000];
	GLfloat circleVerticesY[1000];
	GLfloat circleVerticesZ[1000];
	GLfloat allCircleVertices[1000];


	//circleVerticesX[0] = x;
	//circleVerticesY[0] = y;
	//circleVerticesZ[0] = z;

	for (int i = 0; i < numberOfVertices; i++)
	{
		circleVerticesX[i] = x + (radius * cos(i * doublePi / numberOfSides));
		circleVerticesY[i] = y + (radius * sin(i * doublePi / numberOfSides));
		circleVerticesZ[i] = z;
	}


	for (int i = 0; i < numberOfVertices; i++)
	{
		allCircleVertices[i * 3] = circleVerticesX[i];
		allCircleVertices[(i * 3) + 1] = circleVerticesY[i];
		allCircleVertices[(i * 3) + 2] = circleVerticesZ[i];
	}

	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, allCircleVertices);
	glDrawArrays(GL_LINE_STRIP, 0, numberOfVertices);
	glDisableClientState(GL_VERTEX_ARRAY);

}

void Draw::drawLine(GLfloat x1, GLfloat y1, GLfloat x2, GLfloat y2)
{
	GLfloat m, vertices[3000], y, x;
	m = (y2 - y1) / (x2 - x1);
	int j = 0;
	/*for (GLfloat i = x1;i <= x2;i+=0.1)
	{
		y = m * i - m * x1 + y1;
		vertices[j++] = i;
		vertices[j++] = y;
		vertices[j++] = 0;
	}*/
	//printf("%d %f\n", j, vertices[j-1]);
	vertices[0] = x1;
	vertices[1] = y1;
	vertices[2] = 0;
	vertices[3] = x2;
	vertices[4] = y2;
	vertices[5] = 0;

	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, vertices);
	glDrawArrays(GL_LINES, 0, 6);
	glDisableClientState(GL_VERTEX_ARRAY);

}

Draw::~Draw()
{

}