#include <cmath>
#include <iostream>
#include "GL\glut.h"
#include "vector3D.h"
#include <chrono>
#include <math.h>

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

class Cube : public RtObject {

	Vector3 near_, far_;

public:
	Cube(Vector3 n, Vector3 f) { near_ = n; far_ = f; };
	double intersectWithRay(Ray ray, Vector3& pos, Vector3& normal);
};

class Plane : public RtObject {
	// the plane of any point v such that v . n = d
	Vector3 n_;
	double d_;
public:
	Plane(Vector3 n, double d) { n_ = n; d_ = d; };
	void set(Vector3 n, double d) { n_ = n; d_ = d; };
	double intersectWithRay(Ray, Vector3& pos,
		Vector3& normal);
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

RtObject **objList; // The list of all objects in the scene


// Global Variables
// Camera Settings
Vector3 cameraPos(0,0,-500);

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

double bgColor[3] = { 0.2,0.1,0.35 };//{ 0.1,0.1,0.4 };

int sceneNo = 0;

double Cube::intersectWithRay(Ray r, Vector3& intersection, Vector3& normal) {

	float t1, t2, tNear = -1000.0f, tFar = 1000.0f, temp, tCube;
	bool intFlag = true;
	
	for (int i = 0; i < 3; i++) {
		if (r.dir.x[i] == 0) {
			if (r.start.x[i] < near_.x[i] || r.start.x[i] > far_.x[i])
				intFlag = false;
		}
		else {
			t1 = (near_.x[i] - r.start.x[i]) / r.dir.x[i];
			t2 = (far_.x[i] - r.start.x[i]) / r.dir.x[i];
			
			if (t1 > t2) {
				temp = t1;
				t1 = t2;
				t2 = temp;
			}
			
			if (t1 > tNear)
				tNear = t1;
			
			if (t2 < tFar)
				tFar = t2;
			
			if (tNear > tFar)
				intFlag = false;
			
			if (tFar < 0)
				intFlag = false;
		}
	}

	if (intFlag == false)
		tCube = -1;
	else {
		tCube = tNear;
		// set intersection
		intersection = r.start + r.dir * tCube;

		// set normal (from intersection to center)
		if (abs(intersection.x[0] - near_.x[0]) < 0.01)
			normal = Vector3(-1, 0, 0);
		else if (abs(intersection.x[1] - near_.x[1]) < 0.01)
			normal = Vector3(0, -1, 0);
		else if (abs(intersection.x[2] - near_.x[2]) < 0.01)
			normal = Vector3(0, 0, -1);
		else if (abs(intersection.x[0] - far_.x[0]) < 0.01)
			normal = Vector3(1, 0, 0);
		else if (abs(intersection.x[1] - far_.x[1]) < 0.01)
			normal = Vector3(0, 1, 0);
		else if (abs(intersection.x[2] - far_.x[2]) < 0.01)
			normal = Vector3(0, 0, 1);
	}

	return tCube;
}

double Plane::intersectWithRay(Ray r, Vector3& intersection, Vector3& normal)
{
	// l(t) = r.start + t * r.dir
	// so l(t) . n = d     r.start.n + t * r.dir.n = d
	// t = (d-r.start dot n)/(r.dir dot n)

	double t = (d_ - dot_prod(r.start, n_)) / (dot_prod(r.dir, n_));

	intersection = r.start + r.dir * t;
	if (dot_prod(r.dir, n_) > 0)
		normal = -n_;
	else
		normal = n_;
	return t;
};

int calQuadEqn(double a, double b, double c, double& x1, double& x2) {
	double delta = b * b - 4 * a*c;
	if (delta <0)
		return 0;
	
	x1 = (-b + sqrt(delta)) / (2 * a);
	x2 = (-b - sqrt(delta)) / (2 * a);
	
	if (x1 < 0 || (x1 > x2 && x2 > 0)) {
		//swap x1 and x2
		double temp = x1;
		x1 = x2;
		x2 = temp;
	}

	return 1;
}

double Sphere::intersectWithRay(Ray r, Vector3& intersection, Vector3& normal)
// return a -ve if there is no intersection. Otherwise, return the smallest postive value of t
{
	// Step 1
	//smaller +ve gives the nearer intersection 
	//solve quadratic equation function 
	double t1, t2;
	double a, b, c;
	a = b = c = 0;
	for (int i = 0; i < 3; i++) {
		a += pow(r.dir.x[i], 2);
		b += 2 * (r.start.x[i] - center_.x[i])*r.dir.x[i];
		c += pow((r.start.x[i] - center_.x[i]), 2);
	}
	c -= pow(r_, 2);

	if (calQuadEqn(a, b, c, t1, t2)) {
		for (int i = 0; i < 3; i++) {
			intersection.x[i] = r.start.x[i] + r.dir.x[i] * t1;
			normal.x[i] = intersection.x[i] - center_.x[i];
		}

		return t1;
	}

	return -1;
}

void rayTrace(Ray ray, double& r, double& g, double& b, int fromObj = -1 ,int level = 0)
{
	// Step 4
	if (level >= MAX_RT_LEVEL)
		return;

	int goBackGround = 1, i = 0;

	Vector3 intersection, normal;
	//Vector3 lightV;
	//Vector3 viewV;
	//Vector3 lightReflectionV;
	//Vector3 rayReflectionV;
	Vector3 V;
	Vector3 L;
	Vector3 R;

	Ray newRay;
	double mint = DBL_MAX, t;


	for (i = 0; i < NUM_OBJECTS; i++)
	{
		if ((t = objList[i]->intersectWithRay(ray, intersection, normal)) > 0)
		{
 			// Step 2 


			// Step 3
			if (i == fromObj)
				continue;
			if (t > mint)
				continue;
			mint = t;

			/*Initialising for PIE*/  
			V = -ray.dir; //going into the direction of the viewer which is opp of the outgoing ray vector which goes out of viewer's eyes
			L = lightPos - intersection;
			
			//normalize all vectors
			V.normalize();
			L.normalize();
			normal.normalize();
			
			R = normal * 2 * dot_prod(normal, L) - L;//The reflected vector
			R.normalize();

			//establish a new ray to trace
			Ray ray2;
			ray2.start = intersection;
			ray2.dir = normal * 2 * dot_prod(normal, V) - V;
			
			//reccurse
			rayTrace(ray2, r, g, b, i, level + 1);

			/*Calculating PIE*/
			r = objList[i]->specularReflection[0] * r //KsIr
				+ (objList[i]->ambiantReflection[0] * ambiantLight[0] //ambient
				+ dot_prod(normal, L) *diffusetLight[0] * objList[i]->diffusetReflection[0] //diffuse 
				+ pow(dot_prod(R, V), objList[i]->speN) * specularLight[0] * objList[i]->specularReflection[0]); //specular
			
			g = objList[i]->specularReflection[1] * g 
				+ (objList[i]->ambiantReflection[1] * ambiantLight[1] 
				+ dot_prod(normal, L) *diffusetLight[1] * objList[i]->diffusetReflection[1] 
				+ pow(dot_prod(R, V), objList[i]->speN) * specularLight[1] * objList[i]->specularReflection[1]);
			
			b = objList[i]->specularReflection[2] * b 
				+ (objList[i]->ambiantReflection[2] * ambiantLight[2] 
				+ dot_prod(normal, L) *diffusetLight[2] * objList[i]->diffusetReflection[2] 
				+ pow(dot_prod(R, V), objList[i]->speN) * specularLight[2] * objList[i]->specularReflection[2]);

			//shadow ray
			Ray shadowRay;
			shadowRay.start = intersection;
			shadowRay.dir = lightPos.operator-(intersection);
			shadowRay.dir.normalize();
			
			for (int j = 0; j < NUM_OBJECTS; j++) {
				if (j != i && (t = objList[j]->intersectWithRay(shadowRay, intersection, normal)) > 0) { // Not the same object and intersecting a different object
					r -= 0.3 * objList[i]->specularReflection[0]; 
					g -= 0.5 * objList[i]->specularReflection[1];
					b -= 0.5 * objList[i]->specularReflection[2];
				}
			}

			//Put a limit to rgb values
			if (r > 1)
				r = 1;
			if (g > 1)
				g = 1;
			if (b > 1)
				b = 1;

			goBackGround = 0;
		}
	}

	if (goBackGround)
	{
		r = bgColor[0];
		g = bgColor[1];
		b = bgColor[2];
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
		objList[2] = new Sphere(Vector3(-130, -80, -80), 100);

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
		objList[2]->specularReflection[0] = 0.3;
		objList[2]->specularReflection[1] = 0.7;
		objList[2]->specularReflection[2] = 0.3;
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
		// Step 5
		//mickey mouse
		/*
		objList[3] = new Plane(Vector3(-50, 10, 1), 50.0);
		((Sphere*)objList[0])->set(Vector3(0, -40, 180), 100);
		((Sphere*)objList[1])->set(Vector3(180, 40, 80), 80);
		((Sphere*)objList[2])->set(Vector3(-130, 40, 80), 80);
		*/
		
		//wormhole
		/*
		objList[3] = new Plane(Vector3(50, 0, -30), 5.00);
		((Sphere*)objList[0])->set(Vector3(-100, -330, 300), 400);
		((Sphere*)objList[1])->set(Vector3(180, 40, -80), 20);
		//((Sphere*)objList[3])->set(Vector3(-130, 40, 80), 80);
		objList[2] = new Cube(Vector3(-30, -30, -30), Vector3(200, 200, 200));
		*/

		((Sphere*)objList[0])->set(Vector3(-300, -250, 0), 380);
		((Sphere*)objList[1])->set(Vector3(110, -80, -100), 60);
		objList[2] = new Cube(Vector3(0, -18, -0), Vector3(250, 150, 0));
		((Sphere*)objList[3])->set(Vector3(130, 50, -220), 30);
		//objList[3] = new Plane(Vector3(50, 30, -43), 5.0);
		
		objList[0]->ambiantReflection[0] = 0.1;
		objList[0]->ambiantReflection[1] = 0.4;
		objList[0]->ambiantReflection[2] = 0.6;
		objList[0]->diffusetReflection[0] = 1;
		objList[0]->diffusetReflection[1] = 1;
		objList[0]->diffusetReflection[2] = 0;
		objList[0]->specularReflection[0] = 0.2;
		objList[0]->specularReflection[1] = 0.4;
		objList[0]->specularReflection[2] = 0.4;
		objList[0]->speN = 300;

		objList[1]->ambiantReflection[0] = 0.6;
		objList[1]->ambiantReflection[1] = 0.6;
		objList[1]->ambiantReflection[2] = 0.2;
		objList[1]->diffusetReflection[0] = 0.5;
		objList[1]->diffusetReflection[1] = 0.1;
		objList[1]->diffusetReflection[2] = 0.1;
		objList[1]->specularReflection[0] = 0.2;
		objList[1]->specularReflection[1] = 0.4;
		objList[1]->specularReflection[2] = 0.6;
		objList[1]->speN = 50;

		objList[2]->ambiantReflection[0] = 0.1;
		objList[2]->ambiantReflection[1] = 0.6;
		objList[2]->ambiantReflection[2] = 0.1;
		objList[2]->diffusetReflection[0] = 0.0;
		objList[2]->diffusetReflection[1] = 0.0;
		objList[2]->diffusetReflection[2] = 0.7;
		objList[2]->specularReflection[0] = 0.3;
		objList[2]->specularReflection[1] = 0.7;
		objList[2]->specularReflection[2] = 0.3;
		objList[2]->speN = 650;

		objList[3]->ambiantReflection[0] = 0.3;
		objList[3]->ambiantReflection[1] = 0.3;
		objList[3]->ambiantReflection[2] = 0.3;
		objList[3]->diffusetReflection[0] = 0.2;
		objList[3]->diffusetReflection[1] = 0.8;
		objList[3]->diffusetReflection[2] = 0.3;
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