// CS3241Lab5.cpp
#include <cmath>
#include <iostream>
#include "vector3D.h"
#include <chrono>
#include <float.h>
#include <random>
#include <algorithm>

/* Include header files depending on platform */
#ifdef _WIN32
#include "glut.h"
#define M_PI 3.14159
#elif __APPLE__
#include <OpenGL/gl.h>
#include <GLUT/GLUT.h>
#elif __linux__

#include <GL/glut.h>

#endif

// Copy of PerlinNoise from https://github.com/sol-prog/Perlin_Noise
class PerlinNoise {
    // The permutation vector
    std::vector<int> p;
public:
    // Initialize with the reference values for the permutation vector
    PerlinNoise();
    // Generate a new permutation vector based on the value of seed
    PerlinNoise(unsigned int seed);
    // Get a noise value, for 2D images z can have any value
    double noise(double x, double y, double z);
private:
    double fade(double t);
    double lerp(double t, double a, double b);
    double grad(int hash, double x, double y, double z);
};

// THIS IS A DIRECT TRANSLATION TO C++11 FROM THE REFERENCE
// JAVA IMPLEMENTATION OF THE IMPROVED PERLIN FUNCTION (see http://mrl.nyu.edu/~perlin/noise/)
// THE ORIGINAL JAVA IMPLEMENTATION IS COPYRIGHT 2002 KEN PERLIN

// I ADDED AN EXTRA METHOD THAT GENERATES A NEW PERMUTATION VECTOR (THIS IS NOT PRESENT IN THE ORIGINAL IMPLEMENTATION)

// Initialize with the reference values for the permutation vector
PerlinNoise::PerlinNoise() {

    // Initialize the permutation vector with the reference values
    p = {
            151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
            8,99,37,240,21,10,23,190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
            35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,
            134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
            55,46,245,40,244,102,143,54, 65,25,63,161,1,216,80,73,209,76,132,187,208, 89,
            18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,
            250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
            189,28,42,223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167,
            43,172,9,129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,
            97,228,251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,
            107,49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
            138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180 };
    // Duplicate the permutation vector
    p.insert(p.end(), p.begin(), p.end());
}

// Generate a new permutation vector based on the value of seed
PerlinNoise::PerlinNoise(unsigned int seed) {
    p.resize(256);

    // Fill p with values from 0 to 255
    std::iota(p.begin(), p.end(), 0);

    // Initialize a random engine with seed
    std::default_random_engine engine(seed);

    // Suffle  using the above random engine
    std::shuffle(p.begin(), p.end(), engine);

    // Duplicate the permutation vector
    p.insert(p.end(), p.begin(), p.end());
}

double PerlinNoise::noise(double x, double y, double z) {
    // Find the unit cube that contains the point
    int X = (int) floor(x) & 255;
    int Y = (int) floor(y) & 255;
    int Z = (int) floor(z) & 255;

    // Find relative x, y,z of point in cube
    x -= floor(x);
    y -= floor(y);
    z -= floor(z);

    // Compute fade curves for each of x, y, z
    double u = fade(x);
    double v = fade(y);
    double w = fade(z);

    // Hash coordinates of the 8 cube corners
    int A = p[X] + Y;
    int AA = p[A] + Z;
    int AB = p[A + 1] + Z;
    int B = p[X + 1] + Y;
    int BA = p[B] + Z;
    int BB = p[B + 1] + Z;

    // Add blended results from 8 corners of cube
    double res = lerp(w, lerp(v, lerp(u, grad(p[AA], x, y, z), grad(p[BA], x-1, y, z)), lerp(u, grad(p[AB], x, y-1, z), grad(p[BB], x-1, y-1, z))),	lerp(v, lerp(u, grad(p[AA+1], x, y, z-1), grad(p[BA+1], x-1, y, z-1)), lerp(u, grad(p[AB+1], x, y-1, z-1),	grad(p[BB+1], x-1, y-1, z-1))));
    return (res + 1.0)/2.0;
}

double PerlinNoise::fade(double t) {
    return t * t * t * (t * (t * 6 - 15) + 10);
}

double PerlinNoise::lerp(double t, double a, double b) {
    return a + t * (b - a);
}

double PerlinNoise::grad(int hash, double x, double y, double z) {
    int h = hash & 15;
    // Convert lower 4 bits of hash into 12 gradient directions
    double u = h < 8 ? x : y,
            v = h < 4 ? y : h == 12 || h == 14 ? x : z;
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

// End of PerlinNoise

using namespace std;

#define WINWIDTH 600
#define WINHEIGHT 400
#define MAX_RT_LEVEL 50
#define NUM_SCENE 2

// lightDx and lightDz need to be multiples of this value
#define LIGHT_SAMPLE_INTERVAL 20
// Number of samples on each axis. Total number of samples is around pi * (DOF_SAMPLE / 2)^2
#define DOF_SAMPLES 11
#define SUBSAMPLE_EPSILON 0.05

#define BUMP_MAP_SIZE 800

#define BUMP_MAPS_TOTAL 4

#define BUMP_PERLIN 0
#define BUMP_WOOD 1
#define BUMP_BANDS 2
#define BUMP_RADIAL 3

float *pixelBuffer = new float[WINWIDTH * WINHEIGHT * 3];

default_random_engine generator;
uniform_real_distribution<double> dist(0.0, 1.0);

// Helper functions
double randBetween(double min, double max) {
    return dist(generator) * (max - min) + min;
}

double randBetween(double max = 1) {
    return randBetween(0, max);
}

// a ray that start with "start" and going in the direction "dir"
class Ray {
public:
    Vector3 start, dir;
};

class RtObject {
public:
    // return a -ve if there is no intersection. Otherwise, return the smallest postive value of t
    virtual double intersectWithRay(Ray, Vector3 &pos, Vector3 &normal) = 0;

    // Materials Properties
    double ambientReflection[3];
    double diffuseReflection[3];
    double specularReflection[3];
    double speN = 300;

    bool isReflective() {
        return this->specularReflection[0] && this->specularReflection[1] && this->specularReflection[2];
    }
};

class Sphere : public RtObject {
    Vector3 center_;
    double r_;
    int map_;
    double bumpiness_;
public:
    Sphere(Vector3 c, double r) {
        center_ = c;
        r_ = r;
    };

    Sphere() = default;

    void set(Vector3 c, double r) {
        set(c, r, -1, 0);
    };

    void set(Vector3 c, double r, int map, double bumpiness) {
        center_ = c;
        r_ = r;
        bumpiness_ = bumpiness;
        map_ = map;
    }

    double intersectWithRay(Ray, Vector3 &intersection, Vector3 &normal) override;
};

class Plane : public RtObject {
    Vector3 n_;
    Vector3 d_;

public:
    Plane() = default;

    void set(Vector3 d, Vector3 n) {
        n_ = n;
        d_ = d;
    };

    double intersectWithRay(Ray, Vector3 &intersection, Vector3 &normal) override;
};

double Plane::intersectWithRay(Ray r, Vector3 &intersection, Vector3 &normal) {
    double t = (dot_prod(d_ - r.start, n_)) / (dot_prod(r.dir, n_));
    intersection = r.start + r.dir * t;
    if (dot_prod(r.dir, n_) > 0)
        normal = -n_;
    else
        normal = n_;
    return t;
}

RtObject **objList; // The list of all objects in the scene


// Global Variables
// Camera Settings
Vector3 cameraPos(0, 0, -500);

// assume the the following two vectors are normalised
Vector3 lookAtDir(0, 0, 1);
Vector3 upVector(0, 1, 0);
Vector3 leftVector(1, 0, 0);
float focalLen = 620;
float apertureFStop = 12;

// Light Settings

Vector3 lightPos(900, 1000, -1500); // Extends out in the xz plane
double ambientLight[3] = {0.4, 0.4, 0.4};
double diffuseLight[3] = {0.7, 0.7, 0.7};
double specularLight[3] = {0.5, 0.5, 0.5};

// Size of the area light with lightPos as the center.
// The light extends out in both positive and negative direction by the amount specified here.
// Both values should be a multiple of LIGHT_SAMPLE_INTERVAL
double lightDx = 200;
double lightDz = 100;

double bgColor[3] = {0.13, 0.17, 0.45};
int nObjects;
int superSample = 1;
bool depthOfField = false;
bool drawShadows = false;

// Perlin noise generator
PerlinNoise pn;
double bumps[BUMP_MAPS_TOTAL][BUMP_MAP_SIZE][BUMP_MAP_SIZE];

int sceneNo = 0;


void fillBumpsMatrix() {
    int i, j;

    for (i = 1; i < BUMP_MAP_SIZE - 1; i++) {
        for (j = 1; j < BUMP_MAP_SIZE - 1; j++) {
            double x = ((double) i / BUMP_MAP_SIZE) * 120;
            double y = ((double) j / BUMP_MAP_SIZE) * 60;

            // Perlin
            bumps[BUMP_PERLIN][i][j] = (pn.noise(x, y, 1) + pn.noise(x, y, 2) + pn.noise(x, y, 3)) / 3;

            // Wood - function taken from https://github.com/sol-prog/Perlin_Noise
            double n = 30 * pn.noise(x / 10, y / 10, 1.5);
            bumps[BUMP_WOOD][i][j] = n - floor(n);

            // Pole to pole bands
            bumps[BUMP_RADIAL][i][j] = floor(sin(x));

            // Longitudenal bands
            bumps[BUMP_BANDS][i][j] = floor(sin(y * 1.2));
        }
    }

    // Mirror the edge pixels
    for (i = 1; i < BUMP_MAP_SIZE - 1; i++) {
        for (j = 0; j < BUMP_MAPS_TOTAL; j++) {
            bumps[j][i][0] = bumps[j][i][BUMP_MAP_SIZE - 2];
            bumps[j][i][BUMP_MAP_SIZE - 1] = bumps[j][i][1];

            bumps[j][0][i] = bumps[j][BUMP_MAP_SIZE - 2][i];
            bumps[j][BUMP_MAP_SIZE - 1][i] = bumps[j][1][i];
        }
    }
}

void getSphereBump(int map, double theta, double phi, double &du, double &dv) {
    int u = 1 + (theta / M_PI) * (BUMP_MAP_SIZE - 2);
    int v = 1 + (phi / (2 * M_PI) + 0.5) * (BUMP_MAP_SIZE - 2);

    du = bumps[map][v][u + 1] - bumps[map][v][u - 1];
    dv = bumps[map][v + 1][u] - bumps[map][v - 1][u];

    // cout << theta << ", " << phi << ": " << du << ", " << dv << endl;
}

// Solve quadratic equation and returns the smallest positive root
double solve(double a, double b, double c) {
    double det = (b * b) - (4 * a * c);
    if (det <= 0) return -1; // Imaginary or singular roots

    double det_sqrt = sqrt(det);
    double x1 = (-b + det_sqrt) / (2 * a);
    double x2 = (-b - det_sqrt) / (2 * a);

    if (x1 < 0 && x2 < 0) return -1; // Both negative

    if (x1 < 0) return x2;
    else if (x2 < 0) return x1;
    return min(x1, x2);
}

// Return a -ve if there is no intersection. Otherwise, return the smallest positive value of t
double Sphere::intersectWithRay(Ray r, Vector3 &intersection, Vector3 &normal) {
    // Step 1 - solving the quadratic equation to find intersection
    Vector3 p_c = r.start - center_;
    double t = solve(dot_prod(r.dir, r.dir),
                     dot_prod(r.dir * 2, p_c),
                     p_c.lengthsqr() - (this->r_ * this->r_));

    if (t < 0) return t;

    intersection = r.start + r.dir * t;
    normal = intersection - center_;

    if (bumpiness_ > 0) {
        // Bump mapping - get theta and phi from cartesian coordinates
        // then calculate orthogonal vectors
        double theta = acos(normal.x[1] / r_);
        double phi = atan2(normal.x[0], normal.x[2]);
        double du, dv;

        Vector3 ou = Vector3(normal.x[0] / normal.x[2], 0, normal.x[1] / r_);
        Vector3 ov = cross_prod(ou, normal);
        ou.normalize();
        ov.normalize();

        getSphereBump(map_, theta, phi, du, dv);
        normal = normal + ou * (du * bumpiness_) + ov * (dv * bumpiness_);
    }

    normal.normalize();
    return t;
}

bool isLightBlocked(Vector3 intersection, double dx, double dz, int currentObj) {
    Vector3 _;
    Ray lightRay;
    lightRay.start = intersection;
    lightRay.dir = (lightPos + leftVector * dx + lookAtDir * dz) - intersection;
    lightRay.dir.normalize();

    bool isBlocked = false;
    for (int s = 0; s < nObjects; s++) {
        if (s == currentObj) continue;

        if (objList[s]->intersectWithRay(lightRay, _, _) > 0) {
            isBlocked = true;
            break;
        }
    }

    return isBlocked;
}

// Return a value between 0 and 1 indicating how much light a pixel is receiving
double getPixelLight(int intersectObj, const Vector3 &point) {
    double light = 0;
    double samples = (lightDx / (LIGHT_SAMPLE_INTERVAL / 2)) * (lightDz / (LIGHT_SAMPLE_INTERVAL / 2));
    int cornersBlocked = 0;

    // Check four corners first...
    cornersBlocked += isLightBlocked(point, -lightDx, -lightDz, intersectObj);
    cornersBlocked += isLightBlocked(point, lightDx, -lightDz, intersectObj);
    cornersBlocked += isLightBlocked(point, -lightDx, lightDz, intersectObj);
    cornersBlocked += isLightBlocked(point, lightDx, lightDz, intersectObj);

    if (cornersBlocked == 0) {
        // ... if none are blocked, then we have a fully lit pixel...
        light = 1;
    } else if (cornersBlocked == 4) {
        // ... and if all are blocked, then we have fully occluded pixel...
        light = 0;
    } else {
        // ... otherwise we have to do shadow ray tracing by evenly sampling the rays
        //     from the rays from the area light to the current position to get a soft shadow
        light = 4 - cornersBlocked;

        for (double dx = -lightDx; dx < lightDx; dx += LIGHT_SAMPLE_INTERVAL) {
            for (double dz = -lightDz; dz < lightDz; dz += LIGHT_SAMPLE_INTERVAL) {
                light += !isLightBlocked(point, dx + LIGHT_SAMPLE_INTERVAL / 2, dz + LIGHT_SAMPLE_INTERVAL / 2, intersectObj);
            }
        }

        light /= (samples + 4);
    }
    return light;
}

void rayTrace(Ray ray, double &r, double &g, double &b, int fromObj = -1, int level = 0) {
    if (level > MAX_RT_LEVEL) return;

    int i = 0, intersectObj = -1;

    Vector3 intersection, normal;
    Vector3 lightV, viewV, lightReflectionV;
    Vector3 lightPosN = Vector3(lightPos);
    lightPosN.normalize();

    // Find the correct object to draw
    Vector3 currentIntersection, currentNormal;
    double minT = DBL_MAX, t;
    for (i = 0; i < nObjects; i++) {
        if (i == fromObj) continue;

        RtObject *pObject = objList[i];
        if ((t = pObject->intersectWithRay(ray, currentIntersection, currentNormal)) > 0) {
            // Check that the current object is nearer to the camera than the previously
            // drawn one
            if (t < minT) {
                minT = t;
                intersectObj = i;
                intersection = currentIntersection;
                normal = currentNormal;
            }
        }
    }

    if (intersectObj == -1) {
        // Did not intersect anything, so we draw the background
        r = bgColor[0];
        g = bgColor[1];
        b = bgColor[2];
    } else {
        double N_L, R_V, R_V_N_L, light;

        RtObject *pObject = objList[intersectObj];

        // Step 2 - assign ambient color
        r = ambientLight[0] * pObject->ambientReflection[0];
        g = ambientLight[1] * pObject->ambientReflection[1];
        b = ambientLight[2] * pObject->ambientReflection[2];

        // Step 3 - Phong illumination equation
        // Calculate light ray
        lightV = (lightPos - intersection);
        lightV.normalize();

        N_L = dot_prod(normal, lightV);
        if (N_L <= 0) goto PHONG_END;

        // - Shadow ray - check if the light ray intersects with any object
        light = drawShadows ? getPixelLight(intersectObj, intersection) : 1;

        if (light <= 0) goto PHONG_END;
        // - Diffuse term
        r += light * N_L * diffuseLight[0] * pObject->diffuseReflection[0];
        g += light * N_L * diffuseLight[1] * pObject->diffuseReflection[1];
        b += light * N_L * diffuseLight[2] * pObject->diffuseReflection[2];

        // - Specular term
        if (!pObject->isReflective()) goto PHONG_END;
        viewV = cameraPos - intersection;
        viewV.normalize();
        lightReflectionV = normal * N_L * 2 - lightPosN;
        lightReflectionV.normalize();

        R_V = dot_prod(lightReflectionV, viewV);
        if (R_V <= 0) goto PHONG_END;
        R_V_N_L = light * pow(R_V, pObject->speN);
        r += R_V_N_L * specularLight[0] * pObject->specularReflection[0];
        b += R_V_N_L * specularLight[1] * pObject->specularReflection[1];
        g += R_V_N_L * specularLight[2] * pObject->specularReflection[2];

        // Step 4 - Recursive ray tracing
        PHONG_END: // Beware of raptors
        if (!pObject->isReflective()) return;

        Vector3 rayReflectionV;
        // r = 2 ( N A dot (-i) ) N + i
        rayReflectionV = normal * (2 * dot_prod(normal, -ray.dir)) + ray.dir;
        rayReflectionV.normalize();

        Ray newRay;
        newRay.dir = rayReflectionV;
        newRay.start = intersection;

        double reflectR, reflectG, reflectB;
        rayTrace(newRay, reflectR, reflectG, reflectB, intersectObj, level + 1);

        r += reflectR * pObject->specularReflection[0];
        g += reflectG * pObject->specularReflection[1];
        b += reflectB * pObject->specularReflection[2];
    }
}


void clearPixelBuffer() {
    for (int i = 0; i < WINWIDTH * WINHEIGHT * 3; i++) pixelBuffer[i] = 0;
}

void drawInPixelBuffer(int x, int y, double r, double g, double b) {
    pixelBuffer[(y * WINWIDTH + x) * 3] += (float) r;
    pixelBuffer[(y * WINWIDTH + x) * 3 + 1] += (float) g;
    pixelBuffer[(y * WINWIDTH + x) * 3 + 2] += (float) b;
}

void renderScene() {
    // Clear buffer
    clearPixelBuffer();

    int x, y, sx, sy;
    Ray ray;
    double r, g, b;

    cout << "Rendering Scene " << sceneNo << " with resolution " << WINWIDTH << "x" << WINHEIGHT;
    long long int time1 = chrono::duration_cast<chrono::milliseconds>(
            chrono::steady_clock::now().time_since_epoch()).count(); // marking the starting time

    Vector3 vpCenter = cameraPos + lookAtDir * focalLen;  // viewplane center
    Vector3 startingPt = vpCenter + leftVector * (-WINWIDTH / 2.0) + upVector * (-WINHEIGHT / 2.0);
    Vector3 currPt;

    // Pre-compute starting vectors for depth of field effect
    vector<Vector3> rayStartingPts;

    if (depthOfField) {
        double diameter = focalLen / apertureFStop;
        double interval = diameter / DOF_SAMPLES;

        for (int rx = 0; rx < DOF_SAMPLES; rx++) {
            for (int ry = 0; ry < DOF_SAMPLES; ry++) {
                double dx = -diameter / 2 + rx * interval;
                double dy = -diameter / 2 + ry * interval;

                if (sqrt(dx*dx + dy*dy) < diameter / 2) {
                    rayStartingPts.push_back(cameraPos + leftVector * dx + upVector * dy);
                }
            }
        }

        cout << endl << "Sampling DOF over " << rayStartingPts.size() << " points" << endl;
    } else {
        rayStartingPts.push_back(cameraPos);
    }

    // Pre-compute super-sampling sample count
    int samples = (superSample * superSample) * rayStartingPts.size();

    for (x = 0; x < WINWIDTH; x++) {
        for (y = 0; y < WINHEIGHT; y++) {
            // Super-sampling - if superSample is larger than 1, we make superSample^2 samples
            // to generate each pixel value, resulting in a smoother image at the expense of
            // significantly more computation
            double d = 1.0 / superSample;

            for (sx = 0; sx < superSample; sx++) {
                for (sy = 0; sy < superSample; sy++) {
                    // Depth of field - sample rays from across a simulated aperture
                    for (auto rayStartPt: rayStartingPts) {
                        currPt = startingPt + leftVector * (x + sx * d) + upVector * (y + sy * d);
                        ray.start = rayStartPt;
                        ray.dir = currPt - rayStartPt;
                        ray.dir.normalize();
                        rayTrace(ray, r, g, b);

                        drawInPixelBuffer(x, y, r / samples, g / samples, b / samples);
                    }
                }
            }
        }
    }

    long long int time2 = chrono::duration_cast<chrono::milliseconds>(
            chrono::steady_clock::now().time_since_epoch()).count(); // marking the ending time

    cout << "  Done! \nRendering time = " << time2 - time1 << "ms" << endl << endl;
}


void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_DOUBLEBUFFER);
    glDrawPixels(WINWIDTH, WINHEIGHT, GL_RGB, GL_FLOAT, pixelBuffer);
    glutSwapBuffers();
    glFlush();
}

void reshape(int w, int h) {
    glViewport(0, 0, (GLsizei) w, (GLsizei) h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glOrtho(-10, 10, -10, 10, -10, 10);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}


void setScene(int i = 0) {
    if (i > NUM_SCENE) {
        cout << "Warning: Invalid Scene Number" << endl;
        return;
    }

    if (i == 0) {
        // Customize properties for this scene
        focalLen = 500;
        nObjects = 4;

        // Create scene objects
        objList[0] = new Sphere();
        objList[1] = new Sphere();
        objList[2] = new Sphere();
        objList[3] = new Sphere();

        ((Sphere *) objList[0])->set(Vector3(-130, 80, 120), 100);
        ((Sphere *) objList[1])->set(Vector3(130, -80, -80), 100);
        ((Sphere *) objList[2])->set(Vector3(-130, -80, -80), 100);
        ((Sphere *) objList[3])->set(Vector3(130, 80, 120), 100);

        objList[0]->ambientReflection[0] = 0.1;
        objList[0]->ambientReflection[1] = 0.4;
        objList[0]->ambientReflection[2] = 0.4;
        objList[0]->diffuseReflection[0] = 0;
        objList[0]->diffuseReflection[1] = 1;
        objList[0]->diffuseReflection[2] = 1;
        objList[0]->specularReflection[0] = 0.2;
        objList[0]->specularReflection[1] = 0.4;
        objList[0]->specularReflection[2] = 0.4;
        objList[0]->speN = 300;

        objList[1]->ambientReflection[0] = 0.6;
        objList[1]->ambientReflection[1] = 0.6;
        objList[1]->ambientReflection[2] = 0.2;
        objList[1]->diffuseReflection[0] = 1;
        objList[1]->diffuseReflection[1] = 1;
        objList[1]->diffuseReflection[2] = 0;
        objList[1]->specularReflection[0] = 0.0;
        objList[1]->specularReflection[1] = 0.0;
        objList[1]->specularReflection[2] = 0.0;
        objList[1]->speN = 50;

        objList[2]->ambientReflection[0] = 0.1;
        objList[2]->ambientReflection[1] = 0.6;
        objList[2]->ambientReflection[2] = 0.1;
        objList[2]->diffuseReflection[0] = 0.1;
        objList[2]->diffuseReflection[1] = 1;
        objList[2]->diffuseReflection[2] = 0.1;
        objList[2]->specularReflection[0] = 0.3;
        objList[2]->specularReflection[1] = 0.7;
        objList[2]->specularReflection[2] = 0.3;
        objList[2]->speN = 650;

        objList[3]->ambientReflection[0] = 0.3;
        objList[3]->ambientReflection[1] = 0.3;
        objList[3]->ambientReflection[2] = 0.3;
        objList[3]->diffuseReflection[0] = 0.7;
        objList[3]->diffuseReflection[1] = 0.7;
        objList[3]->diffuseReflection[2] = 0.7;
        objList[3]->specularReflection[0] = 0.6;
        objList[3]->specularReflection[1] = 0.6;
        objList[3]->specularReflection[2] = 0.6;
        objList[3]->speN = 650;

    }

    if (i == 1) {
        // Customize properties for this scene
        focalLen = 570;
        nObjects = 7;
        double floorY = -160;

        // Create scene objects
        objList[0] = new Sphere();
        objList[1] = new Sphere();
        objList[2] = new Sphere();
        objList[3] = new Sphere();
        objList[4] = new Plane();
        objList[5] = new Sphere();
        objList[6] = new Sphere();

        ((Sphere *) objList[0])->set(Vector3(-150, floorY + 140, 300), 140, BUMP_WOOD, 20);
        ((Sphere *) objList[1])->set(Vector3(130, floorY + 80 + 150, -80), 80, BUMP_BANDS, 20);
        ((Sphere *) objList[2])->set(Vector3(-30, floorY + 40, -60), 40, BUMP_RADIAL, 10);
        ((Sphere *) objList[3])->set(Vector3(60, floorY + 85, 120), 85, BUMP_WOOD, 15);
        ((Plane *) objList[4])->set(Vector3(130, -160, 120), Vector3(0, 1, 0));
        ((Sphere *) objList[5])->set(Vector3(20, floorY + 35 + 210, 60), 35);
        ((Sphere *) objList[6])->set(Vector3(140, floorY + 35, 35), 35);

        // Big ball behind
        objList[0]->ambientReflection[0] = 0.1;
        objList[0]->ambientReflection[1] = 0.4;
        objList[0]->ambientReflection[2] = 0.3;
        objList[0]->diffuseReflection[0] = 0.1;
        objList[0]->diffuseReflection[1] = 0.8;
        objList[0]->diffuseReflection[2] = 0.8;
        objList[0]->specularReflection[0] = 0.4;
        objList[0]->specularReflection[1] = 0.4;
        objList[0]->specularReflection[2] = 0.4;
        objList[0]->speN = 300;

        // Big, almost non-reflective ball raised above the surface
        objList[1]->ambientReflection[0] = 0.6;
        objList[1]->ambientReflection[1] = 0.5;
        objList[1]->ambientReflection[2] = 0.0;
        objList[1]->diffuseReflection[0] = 0.7;
        objList[1]->diffuseReflection[1] = 0.8;
        objList[1]->diffuseReflection[2] = 0.0;
        objList[1]->specularReflection[0] = 0.085;
        objList[1]->specularReflection[1] = 0.085;
        objList[1]->specularReflection[2] = 0.085;
        objList[1]->speN = 50;

        // Small ball in front and center
        objList[2]->ambientReflection[0] = 0.1;
        objList[2]->ambientReflection[1] = 0.3;
        objList[2]->ambientReflection[2] = 0.1;
        objList[2]->diffuseReflection[0] = 0.13;
        objList[2]->diffuseReflection[1] = 0.8;
        objList[2]->diffuseReflection[2] = 0.18;
        objList[2]->specularReflection[0] = 0.69;
        objList[2]->specularReflection[1] = 0.7;
        objList[2]->specularReflection[2] = 0.16;
        objList[2]->speN = 650;

        // Medium sized shiny ball on right
        objList[3]->ambientReflection[0] = 0.3;
        objList[3]->ambientReflection[1] = 0.3;
        objList[3]->ambientReflection[2] = 0.3;
        objList[3]->diffuseReflection[0] = 0.7;
        objList[3]->diffuseReflection[1] = 0.7;
        objList[3]->diffuseReflection[2] = 0.7;
        objList[3]->specularReflection[0] = 0.6;
        objList[3]->specularReflection[1] = 0.6;
        objList[3]->specularReflection[2] = 0.6;
        objList[3]->speN = 650;

        // Shiny mirror like surface on the floor
        objList[4]->ambientReflection[0] = 0.7;
        objList[4]->ambientReflection[1] = 0.7;
        objList[4]->ambientReflection[2] = 0.7;
        objList[4]->diffuseReflection[0] = 0.9;
        objList[4]->diffuseReflection[1] = 0.9;
        objList[4]->diffuseReflection[2] = 0.9;
        objList[4]->specularReflection[0] = 0.3;
        objList[4]->specularReflection[1] = 0.3;
        objList[4]->specularReflection[2] = 0.3;
        objList[4]->speN = 650;

        // Small sized shiny ball above objectList[3]
        objList[5]->ambientReflection[0] = 0.5;
        objList[5]->ambientReflection[1] = 0.2;
        objList[5]->ambientReflection[2] = 0.2;
        objList[5]->diffuseReflection[0] = 0.8;
        objList[5]->diffuseReflection[1] = 0.7;
        objList[5]->diffuseReflection[2] = 0.7;
        objList[5]->specularReflection[0] = 0.4;
        objList[5]->specularReflection[1] = 0.3;
        objList[5]->specularReflection[2] = 0.3;
        objList[5]->speN = 650;

        // Small sized shiny ball in front of objectList[3]
        objList[6]->ambientReflection[0] = 0.5;
        objList[6]->ambientReflection[1] = 0.2;
        objList[6]->ambientReflection[2] = 0.2;
        objList[6]->diffuseReflection[0] = 0.8;
        objList[6]->diffuseReflection[1] = 0.7;
        objList[6]->diffuseReflection[2] = 0.7;
        objList[6]->specularReflection[0] = 0.4;
        objList[6]->specularReflection[1] = 0.3;
        objList[6]->specularReflection[2] = 0.3;
        objList[6]->speN = 650;
    }
}

void keyboard(unsigned char key, int x, int y) {
    switch (key) {
        case 's':
        case 'S':
            sceneNo = (sceneNo + 1) % NUM_SCENE;
            setScene(sceneNo);
            renderScene();
            glutPostRedisplay();
            break;

        case 'h':
        case 'H':
            drawShadows = !drawShadows;
            renderScene();
            glutPostRedisplay();
            break;

        case 'd':
        case 'D':
            depthOfField = !depthOfField;
            renderScene();
            glutPostRedisplay();
            break;

        case 'a':
        case 'A':
            superSample = superSample == 1 ? 2 : 1;
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

int main(int argc, char **argv) {
    cout << "<<CS3241 Lab 5>>\n\n" << endl;
    cout << "S to go to next scene" << endl;
    cout << "H to toggle shadows" << endl;
    cout << "A to toggle 2x2 super sampling" << endl;
    cout << "D to toggle depth of field (very slow)" << endl;
    cout << "Q to quit" << endl;
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(WINWIDTH, WINHEIGHT);

    glutCreateWindow("CS3241 Lab 5: Ray Tracing");

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);

    glutKeyboardFunc(keyboard);

    objList = new RtObject *[20];
    fillBumpsMatrix();

    setScene(0);

    setScene(sceneNo);
    renderScene();

    glutMainLoop();

    for (int i = 0; i < nObjects; i++)
        delete objList[i];
    delete[] objList;

    delete[] pixelBuffer;

    return 0;
}