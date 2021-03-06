// EmojiMaker.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

//#include "pch.h"
#include <iostream>
#include <GL/glew.h> 
#include <GLFW/glfw3.h>
#include <math.h>
#include <stdio.h>
#include "Happy.h"
#include "Angry.h"
#include "Surprise.h"
#include "Disgust.h"
#include "Sad.h"
#include "Neutral.h"
#include "Fear.h"
#include <fstream>
using namespace std;

#define SCREEN_WIDTH 1240
#define SCREEN_HEIGHT 480
#define M_PI 3.14159265

int main(void)
{

	Happy khushi;
	Angry gussa;
	Surprise achambhit;
	Disgust ghrna;
	Sad dukhi;
	Neutral tatasth;
	Fear darr;

	GLFWwindow *window, *window2;
	// Initialize the library
	if (!glfwInit())
	{
		return -1;
	}


	window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "EMOJI ^_^  RUN", NULL, NULL);



	if (!window)
	{
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);

	glViewport(0.0f, 0.0f, SCREEN_WIDTH, SCREEN_HEIGHT);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, SCREEN_WIDTH, 0, SCREEN_HEIGHT, 0, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();


	while (!glfwWindowShouldClose(window))
	{
		glClear(GL_COLOR_BUFFER_BIT);
		//glfwMaximizeWindow(window);
		char ch[1];
		ifstream fin("C:/Users/ashyo98/Desktop/testfile.txt");
		fin >> ch;
		fin.close();
		int value = ch[0] - '0';
		switch (value)
		{
		case 0: 
			gussa.angryFace_1();
			gussa.angryFace_2(250);
			gussa.angryFace_3(510);
			glfwSetWindowTitle(window, u8"Angry ('A')");
			break;
		case 1:
			ghrna.disgustface_1();
			ghrna.disgustface_2(260);
			ghrna.disgustface_3(550);
			glfwSetWindowTitle(window, u8"Disgust ( -.-)");
			break;
		case 2:
			darr.fearFace_1();
			darr.fearFace_2(250);
			darr.fearFace_3(500);
			glfwSetWindowTitle(window, u8"Fear (O∆O)");
			break;
		case 3:
			khushi.happyFace_1();
			khushi.happyFace_2(250);
			khushi.happyFace_3(490);
			glfwSetWindowTitle(window, u8"Happy (•‿•)");
			break;
		case 4: 
			dukhi.sadFace_1();
			dukhi.sadFace_2(250);
			dukhi.sadFace_3(500);
			glfwSetWindowTitle(window, u8"Sad ●︿●");
			break;
		case 5:
			achambhit.SurpriseFace_1();
			achambhit.SurpriseFace_2(500);
			glfwSetWindowTitle(window, u8"Surprise ⊙０⊙");
			break;
		case 6:
			tatasth.neutralFace_1();
			tatasth.neutralFace_2(240);
			tatasth.neutralFace_3(500);
			glfwSetWindowTitle(window, u8"Neutral -_-");
			break;
		default:
				printf("Emotionless\n");
		}

		// ---->>>>Swap front and back buffers
		glfwSwapBuffers(window);

		// ----->>>Poll for and process events
		glfwPollEvents();
	}

	glfwTerminate();

	return 0;
}

