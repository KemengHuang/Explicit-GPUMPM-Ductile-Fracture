#include <GL\glew.h>
#include <GL\freeglut.h>
#include <sstream>
#include"glmainHead.h"
#include<fstream>
#include"Simulator.h"
#include "cuda_runtime.h"
#include "gl_texture.h"
MPMSimulator simulatorMPM;

int framest = 0;
char* window_title;

GLuint v;
GLuint f;
GLuint p;


Vector4DF	light[2], light_to[2];				// Light stuff

static const std::string scale_density_filename = "scale_d.txt";
static const std::string scale_force_filename = "scale_f.txt";

bool screenshot = false;
bool stop = true;
float detaTime = 1000.f;
float preTime = 0;
float simuTime = 0;
float cflTime = 0;

void init_cuda() {
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		exit(0);
	}
}


void draw_box(float ox, float oy, float oz, float width, float height, float length)
{
	glLineWidth(2.5f);
	glColor3f(0.8f, 0.8f, 0.8f);

	glBegin(GL_LINES);

	glVertex3f(ox, oy, oz);
	glVertex3f(ox + width, oy, oz);

	glVertex3f(ox, oy, oz);
	glVertex3f(ox, oy + height, oz);

	glVertex3f(ox, oy, oz);
	glVertex3f(ox, oy, oz + length);

	glVertex3f(ox + width, oy, oz);
	glVertex3f(ox + width, oy + height, oz);

	glVertex3f(ox + width, oy + height, oz);
	glVertex3f(ox, oy + height, oz);

	glVertex3f(ox, oy + height, oz + length);
	glVertex3f(ox, oy, oz + length);

	glVertex3f(ox, oy + height, oz + length);
	glVertex3f(ox, oy + height, oz);

	glVertex3f(ox + width, oy, oz);
	glVertex3f(ox + width, oy, oz + length);

	glVertex3f(ox, oy, oz + length);
	glVertex3f(ox + width, oy, oz + length);

	glVertex3f(ox + width, oy + height, oz);
	glVertex3f(ox + width, oy + height, oz + length);

	glVertex3f(ox + width, oy + height, oz + length);
	glVertex3f(ox + width, oy, oz + length);

	glVertex3f(ox, oy + height, oz + length);
	glVertex3f(ox + width, oy + height, oz + length);

	glEnd();
}

//sf 设置长方体
void init_mpm_system()
{
	//T dt = 0;
	//unsigned int typem = 0;
	//std::cin >> typem;
	simulatorMPM.build(typem);
	//simulatorMPM.simulateStick(&dt);
}
GLuint position_vbo_;
GLuint color_vbo_;
PNGTexture particle_texture_;
void init()
{
	init_cuda();
	glewInit();

	glViewport(0, 0, window_width, window_height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(45.0, (float)window_width / window_height, 10.0f, 500.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0f, 0.0f, -3.0f);

	light[0].x = 0.5;		light[0].y = 1.5;	light[0].z = 0.5; light[0].w = 1;
	light_to[0].x = 0;	light_to[0].y = 0;	light_to[0].z = 0; light_to[0].w = 1;

	light[1].x = 55;		light[1].y = 140;	light[1].z = 50;	light[1].w = 1;
	light_to[1].x = 0;	light_to[1].y = 0;	light_to[1].z = 0;		light_to[1].w = 1;

	real_world_origin.x = 0; real_world_origin.y = 0; real_world_origin.z = 0;
	real_world_side.x = 1; real_world_side.y = 1; real_world_side.z = 1;
	
	auto assets_path = std::string{ gmpm_ASSETS_DIR };
	assets_path = assets_path + "sample/ball32.png";
	particle_texture_.loadPNG(assets_path.c_str());
	glGenBuffers(1, &position_vbo_);
	glGenBuffers(1, &color_vbo_);
}

void init_ratio()
{
	//sim_ratio = real_world_side / sph->hParam->world_size;
}

typedef unsigned int uint;

std::vector<uint> color;
void drawParticles(int step) {
	glEnable(GL_BLEND);
	glEnable(GL_ALPHA_TEST);
	glAlphaFunc(GL_GREATER, 0.5);
	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_COLOR_MATERIAL);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	glEnable(GL_POINT_SPRITE_ARB);
	float quadratic[] = { 1.0f, 0.01f, 0.001f };
	glEnable(GL_POINT_DISTANCE_ATTENUATION);
	glPointParameterfvARB(GL_POINT_DISTANCE_ATTENUATION, quadratic);
	glPointSize(3);
	glPointParameterfARB(GL_POINT_SIZE_MAX, 32);
	glPointParameterfARB(GL_POINT_SIZE_MIN, 1.0f);

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, particle_texture_.get_texture());
	glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	int nump_ = simulatorMPM.numParticle;
	//if (step == 0) {
	//	glGenBuffers(1, &position_vbo_);
	//	glGenBuffers(1, &color_vbo_);
	//	//pos = new mFloat3[nump_];
	//	//color = new uint[nump_];
	//	/*for (unsigned int i = 0; i < nump_; ++i)
	//	{
	//		uint clo = (uint((1.0) * 255.0f) << 24) | (uint((0.7) * 255.0f) << 16) | (uint((0.55) * 255.0f) << 8) | uint((0.15) * 255.0f);
	//		color.push_back(clo);
	//	}*/
	//}
	// Point buffers
	glBindBuffer(GL_ARRAY_BUFFER, position_vbo_);
	glBufferData(GL_ARRAY_BUFFER, nump_ * sizeof(vector3T), &(simulatorMPM.h_pos[0]), GL_DYNAMIC_DRAW);
	if (sizeof(T) == 4) {
		glVertexPointer(3, GL_FLOAT, 0, 0x0);
	}
	else if (sizeof(T) == 8) {
		glVertexPointer(3, GL_DOUBLE, 0, 0x0);
	}
	
	glBindBuffer(GL_ARRAY_BUFFER, color_vbo_);
	glBufferData(GL_ARRAY_BUFFER, nump_ * sizeof(uint), simulatorMPM.getModelParticlePtr(0)->h_color, GL_DYNAMIC_DRAW);
	glColorPointer(4, GL_UNSIGNED_BYTE, 0, 0x0);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);

	// Render - Point Sprites
	glNormal3f(0, 1, 0.001);
	glColor4f(1, 1, 1, 1);
	glDrawArrays(GL_POINTS, 0, nump_);
	// Restore state
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
	glDisable(GL_POINT_SPRITE_ARB);
	glDisable(GL_ALPHA_TEST);
	glDisable(GL_TEXTURE_2D);
	glDepthMask(GL_TRUE);
}
void setOrthographicProjection(GLdouble w, GLdouble h)
{
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0, w, h, 0);
	glMatrixMode(GL_MODELVIEW);
}
void restorePerspectiveProjection()
{
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
}
void renderBitmapString(float x, float y, float z, void* font, const std::stringstream& ss)
{
	std::string str = ss.str();
	const char* c;
	glRasterPos3f(x, y, z);
	for (c = str.c_str(); *c != '\0'; c++) {
		glutBitmapCharacter(font, *c);
	}
}
int countF = 0;

void drawInfo(GLdouble w, GLdouble h)
{
	float x = 20, y = 20, delta_y = 20;
	std::stringstream ss;
	static unsigned int frame = 0;
	static float time = 0.0f, acc_time = 0.0f;

	setOrthographicProjection(w, h);

	glPushMatrix();
	glLoadIdentity();
	glColor3f(1.0f, 1.0f, 1.0f);
	glDisable(GL_LIGHTING);

	ss << "FPS: " << 1000 / detaTime<<std::endl;
	renderBitmapString(x, y, 0, GLUT_BITMAP_HELVETICA_12, ss);
	ss.str(""); y += delta_y;

	ss << "cflTime: " << cflTime << std::endl;
	renderBitmapString(x, y, 0, GLUT_BITMAP_HELVETICA_12, ss);
	ss.str(""); y += delta_y;

	ss << "PreprocessTime: " << preTime << std::endl;
	renderBitmapString(x, y, 0, GLUT_BITMAP_HELVETICA_12, ss);
	ss.str(""); y += delta_y;

	ss << "SimulationTime: " << simuTime;
	renderBitmapString(x, y, 0, GLUT_BITMAP_HELVETICA_12, ss);
	ss.str(""); y += delta_y;


	glPopMatrix();

	restorePerspectiveProjection();
}

int step = 0;
void draw_scene()
{
	glPushMatrix();
	glTranslatef(xTrans, yTrans, zTrans);
	glRotatef(xRot, 1.0f, 0.0f, 0.0f);
	glRotatef(yRot, 0.0f, 1.0f, 0.0f);
	glTranslatef(-0.5, -0.5, -0.5);
	// draw framework
	glDisable(GL_LIGHTING);
	draw_box(real_world_origin.x, real_world_origin.y, real_world_origin.z, real_world_side.x, real_world_side.y, real_world_side.z);

	//draw_page(real_world_origin.x, real_world_origin.y, real_world_origin.z, real_world_side.x, real_world_side.y, real_world_side.z);

	//draw light
	draw_box(light[0].x, light[0].y, light[0].z, 0.005f, 0.005f, 0.005f);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glDisable(GL_COLOR_MATERIAL);

	Vector4DF amb, diff, spec;
	float shininess = 5.0;

	float pos[4];
	pos[0] = light[0].x;
	pos[1] = light[0].y;
	pos[2] = light[0].z;
	pos[3] = 1;
	amb.Set(0, 0, 0, 1); diff.Set(1, 1, 1, 1); spec.Set(1, 1, 1, 1);
	glLightfv(GL_LIGHT0, GL_POSITION, (float*)&pos[0]);
	glLightfv(GL_LIGHT0, GL_AMBIENT, (float*)&amb.x);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, (float*)&diff.x);
	glLightfv(GL_LIGHT0, GL_SPECULAR, (float*)&spec.x);

	//GLfloat spot_cutoff = 70.0f;
	//GLfloat spot_pos[] = {0, -1.0, 0};
	//glLightfv(GL_LIGHT0, GL_SPOT_CUTOFF, &spot_cutoff);
	//glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, spot_pos);

	amb.Set(0, 0, 0, 1); diff.Set(.3, .3, .3, 1); spec.Set(.1, .1, .1, 1);
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (float*)&amb.x);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (float*)&diff.x);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (float*)&spec.x);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, (float*)&shininess);


	drawParticles(step++);
	drawInfo(window_width, window_height);

	glPopMatrix();
}

void display_func()
{

	if (!stop) {
		//simulator.simulateStick(&detaTime);
		//simulatorMPM.simulateStick(&detaTime);
		simulatorMPM.simulateStick(&cflTime, &preTime, &simuTime, typem);
		detaTime = preTime + simuTime + cflTime;
		
		countF++;
	}
	framest++;
	glEnable(GL_DEPTH_TEST);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);   //sf 背景颜色
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glDisable(GL_CULL_FACE);
	glShadeModel(GL_SMOOTH);
	glMatrixMode(GL_MODELVIEW);

	draw_scene();

	glutSwapBuffers();
}

void idle_func()
{
	glutPostRedisplay();
}

void reshape_func(GLint width, GLint height)
{
	window_width = width;
	window_height = height;

	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(45.0, (float)width / height, 0.001, 500.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0f, 0.0f, -3.0f);
}

void keyboard_func(unsigned char key, int x, int y)
{
	if (key == 't')
	{
		//		sph_system->tick();

				//for (size_t i = 0; i < 4; ++i)
				//{
				//    float3 pos = sph->get_position(i);
				//    printf("(%f, %f, %f) \n", pos.x, pos.y, pos.z);
				//}
				//printf("\n");
		printf("--------\n");
	}

	if (key == ' ')
	{
		stop = !stop;
	}

	if (key == 'w')
	{
		zTrans += .01f;
	}

	if (key == 's')
	{
		zTrans -= .01f;
	}

	if (key == 'a')
	{
		xTrans += .01f;
	}

	if (key == 'd')
	{
		xTrans -= .01f;
	}

	if (key == 'q')
	{
		yTrans -= .01f;
	}

	if (key == 'e')
	{
		yTrans += .01f;
	}

	if (key == '1')
	{
		//		sph_system->insertParticles(1);
	}

	if (key == '/')
	{
		screenshot = !screenshot;
	}

	glutPostRedisplay();
}

void special_keyboard_func(int key, int x, int y)
{
	switch (key)
	{
	case GLUT_KEY_UP:
		light[0].z -= 0.1f;
		break;
	case GLUT_KEY_DOWN:
		light[0].z += 0.1f;
		break;
	case GLUT_KEY_LEFT:
		light[0].x -= 0.1f;
		break;
	case GLUT_KEY_RIGHT:
		light[0].x += 0.1f;
		break;
	case GLUT_KEY_PAGE_UP:
		light[0].y += 0.1f;
		break;
	case GLUT_KEY_PAGE_DOWN:
		light[0].y -= 0.1f;
		break;
	default:
		break;
	}

	printf("light pos: %f, %f, %f\n", light[0].x, light[0].y, light[0].z);

	glutPostRedisplay();
}

void mouse_func(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		buttonState = 1;
	}
	else if (state == GLUT_UP)
	{
		buttonState = 0;
	}

	ox = x; oy = y;

	glutPostRedisplay();
}

void motion_func(int x, int y)
{
	float dx, dy;
	dx = (float)(x - ox);
	dy = (float)(y - oy);

	if (buttonState == 1)
	{
		xRot += dy / 5.0f;
		yRot += dx / 5.0f;
	}

	ox = x; oy = y;

	glutPostRedisplay();
}

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(0, 0);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("GPUMPM_dragon");

	//	if (!init_cuda()) return -1;
	init();
	init_mpm_system();
	init_ratio();
	
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
	glEnable(GL_POINT_SPRITE_ARB);
	glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);

	glutDisplayFunc(display_func);
	glutReshapeFunc(reshape_func);
	glutKeyboardFunc(keyboard_func);
	glutSpecialFunc(special_keyboard_func);
	glutMouseFunc(mouse_func);
	glutMotionFunc(motion_func);
	glutIdleFunc(idle_func);


	glutMainLoop();

	return 0;
}
