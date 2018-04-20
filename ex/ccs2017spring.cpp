// CS3241Lab5.cpp 
#include <cmath>
#include <iostream>
#include "GL\glut.h"
#include "vector3D.h"
#include <chrono>
#include <algorithm>

using namespace std;

#define WINWIDTH 600
#define WINHEIGHT 400
#define NUM_OBJECTS 4
#define MAX_RT_LEVEL 50
#define NUM_SCENE 2

float* pixelBuffer = new float[WINWIDTH * WINHEIGHT * 3];

class Ray { // a ray that start with "start" and going in the direction "dir"
public:
	Vector3 start, dir;
};

class RtObject {

public:
	virtual double intersectWithRay(Ray, Vector3& pos, Vector3& normal) = 0; // return a -ve if there is no intersection. Otherwise, return the smallest postive value of t

	// Materials Properties
	double ambiantReflection[3] ;
	double diffusetReflection[3] ;
	double specularReflection[3] ;
	double speN = 300;


};

class Sphere : public RtObject {

	Vector3 center_;
	double r_;
public:
	Sphere(Vector3 c, double r) { center_ = c; r_ = r; };
	Sphere() {};
	void set(Vector3 c, double r) { center_ = c; r_ = r; };
	double intersectWithRay(Ray, Vector3& pos, Vector3& normal);
};


class Cylinder : public RtObject {
	Vector3 center_;
	double h_;
	double r_;
public:
	Cylinder(Vector3 c, double h, double r) { center_ = c; h_ = h; r_ = r; };
	Cylinder() {};
	void set(Vector3 c, double h, double r) { center_ = c; h_ = h; r_ = r; };
	double intersectWithRay(Ray, Vector3& pos, Vector3& normal);
};


RtObject **objList; // The list of all objects in the scene


// Global Variables
// Camera Settings
Vector3 cameraPos(0, 0, -500);

// assume the the following two vectors are normalised
Vector3 lookAtDir(0,0,1);
Vector3 upVector(0,1,0);
Vector3 leftVector(1, 0, 0);
float focalLen = 500;

// Light Settings

Vector3 lightPos(900,1000,-1500);
double ambiantLight[3] = { 0.4,0.4,0.4 };
double diffusetLight[3] = { 0.7,0.7, 0.7 };
double specularLight[3] = { 0.5,0.5, 0.5 };


double bgColor[3] = { 0.1,0.1,0.4 };

int sceneNo = 0;


double Sphere::intersectWithRay(Ray r, Vector3& intersection, Vector3& normal)
// return a -ve if there is no intersection. Otherwise, return the smallest postive value of t
{
	Vector3 r0 = r.start;
	Vector3 D = r.dir;
	Vector3 R0 = center_;
	double q = r_;
	
	double alpha = dot_prod(D, D);
	double beta = dot_prod(D * 2, r0 - R0);
	double gamma = dot_prod(R0, R0) + dot_prod(r0, r0) - dot_prod(r0, R0) * 2 - q * q;
	double determinant = beta * beta - 4 * alpha * gamma;
	if (determinant > 0) {
		double x1 = (-beta + sqrt(determinant)) / (2.0 * alpha);
		double x2 = (-beta - sqrt(determinant)) / (2.0 * alpha);

		double t;
		if (x1 > 0 && x2 >0) t = min(x1, x2);
		else if (x1 > 0) t = x1;
		else if (x2 > 0) t = x2;
		else return -1;

		intersection = r0 + (D * t);
		normal = intersection - R0;	
		normal.normalize();
		return t;
	}

	return -1;
}

double Cylinder::intersectWithRay(Ray r, Vector3& intersection, Vector3& normal)
{
	Vector3 r0 = r.start;
	Vector3 D = r.dir;
	Vector3 R0 = center_;

	double alpha = D.x[0] * D.x[0] + D.x[2] * D.x[2];
	double beta = 2 * (r0.x[0] - R0.x[0]) * D.x[0] + 2 * (r0.x[2] - R0.x[2]) * D.x[2];
	double gamma = (r0.x[0] - R0.x[0]) * (r0.x[0] - R0.x[0]) + (r0.x[2] - R0.x[2]) * (r0.x[2] - R0.x[2]) - r_ * r_;
	double ymax = R0.x[1] + h_ / 2;
	double ymin = R0.x[1] - h_ / 2;

	double determinant = beta * beta - 4 * alpha * gamma;
	if (determinant > 0) {
		double x1 = (-beta + sqrt(determinant)) / (2 * alpha);
		double x2 = (-beta - sqrt(determinant)) / (2 * alpha);

		double t;
		double z1 = r0.x[1] + (D.x[1] * x1);
		double z2 = r0.x[1] + (D.x[1] * x2);

		if (ymin <= z1 && z1 <= ymax && ymin <= z2 && z2 <= ymax) {
			if (x1 > 0 && x2 > 0) t = min(x1, x2);
			else if (x1 > 0) t = x1;
			else if (x2 > 0) t = x2;
			else return -1;
		}
		else if (ymin <= z1 && z1 <= ymax) {
			if (x1 > 0) t = x1;
			else return -1;
		}
		else if (ymin <= z2 && z2 <= ymax) {
			if (x2 > 0) t = x2;
			else return -1;
		}
		else {
			return -1;
		}
		
		intersection = r0 + (D * t);
		normal = Vector3(intersection.x[0] - R0.x[0], 0, intersection.x[2] - R0.x[2]);
		normal.normalize();
		
		return t;
	}

	return -1;
}

void rayTrace(Ray ray, double& r, double& g, double& b, int fromObj = -1 ,int level = 0)
{
	if (level > MAX_RT_LEVEL) return;

	int goBackGround = 1, i = 0;

	Vector3 intersection, normal;
	Vector3 lightV; //L
	Vector3 viewV; //V
	Vector3 lightReflectionV; //R
	Vector3 rayReflectionV;

	Ray newRay;
	double mint = DBL_MAX, t;

	double tempr, tempg, tempb;
	int obj;
	for (i = 0; i < NUM_OBJECTS; i++)
	{
		int j = 0;
		if ((t = objList[i]->intersectWithRay(ray, intersection, normal)) > 0)
		{
			if (i == fromObj) continue;

			if (t < mint) mint = t;
			else break;

			lightV = lightPos - intersection;
			lightV.normalize();

			rayReflectionV = normal * 2 * dot_prod(normal, -ray.dir) + ray.dir;
			newRay.dir = rayReflectionV;
			newRay.start = intersection;

			double NdotL = dot_prod(normal, lightV); // make sure less than 1
			lightReflectionV = normal * 2 * dot_prod(normal, lightV) - lightV;
			lightReflectionV.normalize();
			viewV = cameraPos - intersection;
			viewV.normalize();
			double RdotV = dot_prod(lightReflectionV, viewV);

			obj = i;

			tempr = ambiantLight[0] * objList[i]->ambiantReflection[0] + objList[i]->diffusetReflection[0] * diffusetLight[0] * NdotL + objList[i]->specularReflection[0] * specularLight[0] * pow(RdotV, objList[i]->speN);
			tempg = ambiantLight[1] * objList[i]->ambiantReflection[1] + objList[i]->diffusetReflection[1] * diffusetLight[1] * NdotL + objList[i]->specularReflection[1] * specularLight[1] * pow(RdotV, objList[i]->speN);
			tempb = ambiantLight[2] * objList[i]->ambiantReflection[2] + objList[i]->diffusetReflection[2] * diffusetLight[2] * NdotL + objList[i]->specularReflection[2] * specularLight[2] * pow(RdotV, objList[i]->speN);
			goBackGround = 0;

			// check shadow ray
			Ray shadowRay;
			shadowRay.dir = lightV;
			shadowRay.start = intersection;
			
			for (j = 0; j < NUM_OBJECTS; j++)
			{
				if (i == j) continue;
				if ((t = objList[j]->intersectWithRay(shadowRay, intersection, normal)) > 0) {
					tempr = ambiantLight[0] * objList[i]->ambiantReflection[0];
					tempg = ambiantLight[1] * objList[i]->ambiantReflection[1];
					tempb = ambiantLight[2] * objList[i]->ambiantReflection[2];
					break;
				}
			}
		}
	}

	if (goBackGround)
	{
			r = bgColor[0];
			g = bgColor[1];
			b = bgColor[2];
	}
	else {
		rayTrace(newRay, r, g, b, obj, level + 1);
		r = specularLight[0] * objList[obj]->specularReflection[0] * r + tempr;
		g = specularLight[0] * objList[obj]->specularReflection[1] * g + tempg;
		b = specularLight[0] * objList[obj]->specularReflection[2] * b + tempb;
	}
}


void drawInPixelBuffer(int x, int y, double r, double g, double b)
{
	pixelBuffer[(y*WINWIDTH + x) * 3] = (float)r;
	pixelBuffer[(y*WINWIDTH + x) * 3 + 1] = (float)g;
	pixelBuffer[(y*WINWIDTH + x) * 3 + 2] = (float)b;
}

void renderScene()
{
	int x, y;
	Ray ray;
	double r, g, b;

	cout << "Rendering Scene " << sceneNo << " with resolution " << WINWIDTH << "x" << WINHEIGHT << "........... ";
	__int64 time1 = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now().time_since_epoch()).count(); // marking the starting time

	ray.start = cameraPos;

	Vector3 vpCenter = cameraPos + lookAtDir * focalLen;  // viewplane center
	Vector3 startingPt = vpCenter + leftVector * (-WINWIDTH / 2.0) + upVector * (-WINHEIGHT / 2.0);
	Vector3 currPt;

	for(x=0;x<WINWIDTH;x++)
		for (y = 0; y < WINHEIGHT; y++)
		{
			currPt = startingPt + leftVector*x + upVector*y;
			ray.dir = currPt-cameraPos;
			ray.dir.normalize();
			rayTrace(ray, r, g, b);
			drawInPixelBuffer(x, y, r, g, b);
		}

	__int64 time2 = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now().time_since_epoch()).count(); // marking the ending time

	cout << "Done! \nRendering time = " << time2 - time1 << "ms" << endl << endl;
}


void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT |GL_DOUBLEBUFFER);
	glDrawPixels(WINWIDTH, WINHEIGHT, GL_RGB, GL_FLOAT, pixelBuffer);
	glutSwapBuffers();
	glFlush ();
} 

void reshape (int w, int h)
{
	glViewport (0, 0, (GLsizei) w, (GLsizei) h);

	glMatrixMode (GL_PROJECTION);
	glLoadIdentity();

	glOrtho(-10, 10, -10, 10, -10, 10);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}


void setScene(int i = 0)
{
	if (i > NUM_SCENE)
	{
		cout << "Warning: Invalid Scene Number" << endl;
		return;
	}

	if (i == 0)
	{
		objList[3] = new Sphere();
		((Sphere*)objList[0])->set(Vector3(-130, 80, 120), 100);
		((Sphere*)objList[1])->set(Vector3(130, -80, -80), 100);
		((Sphere*)objList[2])->set(Vector3(-130, -80, -80), 100);
		((Sphere*)objList[3])->set(Vector3(130, 80, 120), 100);

		objList[0]->ambiantReflection[0] = 0.1;
		objList[0]->ambiantReflection[1] = 0.4;
		objList[0]->ambiantReflection[2] = 0.4;
		objList[0]->diffusetReflection[0] = 0;
		objList[0]->diffusetReflection[1] = 1;
		objList[0]->diffusetReflection[2] = 1;
		objList[0]->specularReflection[0] = 0.2;
		objList[0]->specularReflection[1] = 0.4;
		objList[0]->specularReflection[2] = 0.4;
		objList[0]->speN = 300;

		objList[1]->ambiantReflection[0] = 0.6;
		objList[1]->ambiantReflection[1] = 0.6;
		objList[1]->ambiantReflection[2] = 0.2;
		objList[1]->diffusetReflection[0] = 1;
		objList[1]->diffusetReflection[1] = 1;
		objList[1]->diffusetReflection[2] = 0;
		objList[1]->specularReflection[0] = 0.0;
		objList[1]->specularReflection[1] = 0.0;
		objList[1]->specularReflection[2] = 0.0;
		objList[1]->speN = 50;

		objList[2]->ambiantReflection[0] = 0.1;
		objList[2]->ambiantReflection[1] = 0.6;
		objList[2]->ambiantReflection[2] = 0.1;
		objList[2]->diffusetReflection[0] = 0.1;
		objList[2]->diffusetReflection[1] = 1;
		objList[2]->diffusetReflection[2] = 0.1;
		objList[2]->specularReflection[0] = 0.1;
		objList[2]->specularReflection[1] = 0.4;
		objList[2]->specularReflection[2] = 0.1;
		objList[2]->speN = 650;

		objList[3]->ambiantReflection[0] = 0.3;
		objList[3]->ambiantReflection[1] = 0.3;
		objList[3]->ambiantReflection[2] = 0.3;
		objList[3]->diffusetReflection[0] = 0.7;
		objList[3]->diffusetReflection[1] = 0.7;
		objList[3]->diffusetReflection[2] = 0.7;
		objList[3]->specularReflection[0] = 0.6;
		objList[3]->specularReflection[1] = 0.6;
		objList[3]->specularReflection[2] = 0.6;
		objList[3]->speN = 650;

	}

	if (i == 1)
	{
		objList[3] = new Cylinder();
		((Sphere*)objList[0])->set(Vector3(200, -80, 180), 100);
		((Sphere*)objList[1])->set(Vector3(-150, 100, 20), 120);
		((Sphere*)objList[2])->set(Vector3(-200, -80, 500), 100);
		((Cylinder*)objList[3])->set(Vector3(0, 0, 300), 500, 85);


		objList[0]->ambiantReflection[0] = 0.1;
		objList[0]->ambiantReflection[1] = 0.4;
		objList[0]->ambiantReflection[2] = 0.4;
		objList[0]->diffusetReflection[0] = 0;
		objList[0]->diffusetReflection[1] = 1;
		objList[0]->diffusetReflection[2] = 1;
		objList[0]->specularReflection[0] = 0.2;
		objList[0]->specularReflection[1] = 0.4;
		objList[0]->specularReflection[2] = 0.4;
		objList[0]->speN = 300;

		objList[1]->ambiantReflection[0] = 0.6;
		objList[1]->ambiantReflection[1] = 0.6;
		objList[1]->ambiantReflection[2] = 0.2;
		objList[1]->diffusetReflection[0] = 1;
		objList[1]->diffusetReflection[1] = 1;
		objList[1]->diffusetReflection[2] = 0;
		objList[1]->specularReflection[0] = 0.0;
		objList[1]->specularReflection[1] = 0.0;
		objList[1]->specularReflection[2] = 0.0;
		objList[1]->speN = 50;

		objList[2]->ambiantReflection[0] = 0.1;
		objList[2]->ambiantReflection[1] = 0.6;
		objList[2]->ambiantReflection[2] = 0.1;
		objList[2]->diffusetReflection[0] = 0.1;
		objList[2]->diffusetReflection[1] = 1;
		objList[2]->diffusetReflection[2] = 0.1;
		objList[2]->specularReflection[0] = 0.1;
		objList[2]->specularReflection[1] = 0.4;
		objList[2]->specularReflection[2] = 0.1;
		objList[2]->speN = 650;

		objList[3]->ambiantReflection[0] = 0.8;
		objList[3]->ambiantReflection[1] = 0.2;
		objList[3]->ambiantReflection[2] = 0.2;
		objList[3]->diffusetReflection[0] = 0.4;
		objList[3]->diffusetReflection[1] = 0.2;
		objList[3]->diffusetReflection[2] = 0.2;
		objList[3]->specularReflection[0] = 0.6;
		objList[3]->specularReflection[1] = 0.6;
		objList[3]->specularReflection[2] = 0.6;
		objList[3]->speN = 650;
	}
} 
  
void keyboard (unsigned char key, int x, int y)
{
	//keys to control scaling - k
	//keys to control rotation - alpha
	//keys to control translation - tx, ty
	switch (key) {
	case 's':
	case 'S':
		sceneNo = (sceneNo + 1 ) % NUM_SCENE;
		setScene(sceneNo);
		renderScene();
		glutPostRedisplay();
		break;
	case 'q':
	case 'Q':
		exit(0);

		default:
		break;
	}
}

int main(int argc, char **argv)
{

	
	cout<<"<<CS3241 Lab 5>>\n\n"<<endl;
	cout << "S to go to next scene" << endl;
	cout << "Q to quit"<<endl;
	glutInit(&argc, argv);
	glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize (WINWIDTH, WINHEIGHT);

	glutCreateWindow ("CS3241 Lab 5: Ray Tracing");

	glutDisplayFunc(display);
	glutReshapeFunc(reshape);

	glutKeyboardFunc(keyboard);

	objList = new RtObject*[NUM_OBJECTS];

	// create four spheres
	objList[0] = new Sphere(Vector3(-130, 80, 120), 100);
	objList[1] = new Sphere(Vector3(130, -80, -80), 100);
	objList[2] = new Sphere(Vector3(-130, -80, -80), 100);
	objList[3] = new Sphere(Vector3(130, 80, 120), 100);

	setScene(0);

	setScene(sceneNo);
	renderScene();

	glutMainLoop();

	for (int i = 0; i < NUM_OBJECTS; i++)
		delete objList[i];
	delete[] objList;

	delete[] pixelBuffer;

	return 0;
}