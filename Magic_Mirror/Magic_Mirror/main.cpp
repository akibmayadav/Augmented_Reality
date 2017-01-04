//Homework2_1
//Ambika Yadav
//Magic Mirror

// References :
// 1. https://github.com/opencv/opencv/blob/master/samples/cpp/dbt_face_detection.cpp

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/gl.h>
#include <GL/glut.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <string>


using namespace std;

cv::VideoCapture *cap = NULL;
int width = 640;
int height = 480;
cv::Mat image;
cv::Mat background_image;
cv::Mat background_image_1;

double fovx,fovy,fx,fy,px,py,aspect,k1,k2,p1,p2,k3,focal_length;
// real dimensions in centimeters
double face_distance ;
double face_length = 15;
double face_width = 10;
int rotation_angle = 0;

cv::Mat cameramatrix;
cv::Mat distortionCoefficients;
cv::Mat glViewMatrix;
cv::Mat viewMatrix=cv::Mat::zeros(4,4,CV_64FC1);

vector<cv::Point3f> objpoint;
vector<cv::Point2f> corners;

string cascade_frontal_name = "/Users/akibmayadav/Documents/OpenCV_Fall2016/Magic_Mirror/haarcascades/haarcascade_frontalface_default.xml";
cv::CascadeClassifier cascade;

cv::Size objimage_smallsize(320,180);
cv::Size objimage_largesize(1280,720);

cv::Size face_smallsize(30,30);
cv::Size face_bigsize(150,150);

cv ::Point3d face_3d_points[5];
cv::Mat tempimage;
cv::Mat gesture_output_image ;

vector<cv::Rect> rectangles;
cv::Rect r1;

//interaction points
bool face_border = false;
bool red_circles = false;
bool tea_pot = false;
bool gesture_1 = false;
bool gesture_2 = false;
bool threshold_scene = false;
int rendering_object_list = 0;

int rotate_iterate()
{
    if(rotation_angle == 360)
        rotation_angle = 0 ;
    else
        rotation_angle+= 3;
    return rotation_angle;
}


void draw_rectangle()
{
    cv::rectangle(tempimage, cv::Point(r1.x*4.0, r1.y*4.0), cv::Point((r1.x + r1.width)*4.0, (r1.y+ r1.height)*4.0), cv::Scalar(0,255,0),2, 8);
    glDrawPixels( tempimage.size().width, tempimage.size().height, GL_BGR, GL_UNSIGNED_BYTE, tempimage.ptr() );
}

void draw_red_circles()
{
    // convert opencv to opengl coordinate system
    glPushMatrix();
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);    // enable lighting
    glEnable(GL_LIGHT0);    // enable
    glEnable(GL_COLOR_MATERIAL); //to enable coloring in lightning
    float lpos[] = { 5, 5, 5, 0 }; // direction and not position
    glLightfv(GL_LIGHT0, GL_POSITION, lpos);
    glShadeModel(GL_SMOOTH);    // smooth shading
    
    for(int i = 0 ; i<5 ; i++)
    {
        glPushMatrix();
        glTranslatef(face_3d_points[i].x, face_3d_points[i].y, -face_3d_points[i].z);
        glColor3d(1,0,0);
        glutSolidSphere(4.0, 100, 100);
        glPopMatrix();
        glFlush();
    }
    glPopMatrix();
    glFlush();
}

void draw_teapot()
{
    glPushMatrix();
    
    double radius = (face_3d_points[1].x-face_3d_points[0].x)+4.0;
    
    // Color Masking
//        glPushMatrix();
//        //glScalef(face_3d_points[1].x- face_3d_points[0].x, face_3d_points[2].y - face_3d_points[0].y,radius*2.0);
//        glColorMask(GL_FALSE,GL_FALSE,GL_FALSE,GL_FALSE);
//        glTranslatef(face_3d_points[4].x,face_3d_points[4].y,-face_3d_points[4].z);
//        glScalef(face_3d_points[1].x- face_3d_points[0].x, face_3d_points[2].y - face_3d_points[0].y,0.1);
//        glutSolidCube(1);
//        glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);
//        glPopMatrix();
    
    // Teapot Rendering
    
    glPushMatrix();
    glTranslatef(face_3d_points[4].x,face_3d_points[4].y,-face_3d_points[4].z);
    
    double object_3d_x = radius*sin(rotate_iterate());
    double object_3d_z = radius*cos(rotate_iterate());
    glTranslatef(object_3d_x, 0.0, -object_3d_z);
    
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);    // enable lighting
    glEnable(GL_LIGHT0);    // enable
    glEnable(GL_COLOR_MATERIAL); //to enable coloring in lightning
    float lpos[] = {5,5,5, 0 }; // direction and not position
    glLightfv(GL_LIGHT0, GL_POSITION, lpos);
    glShadeModel(GL_SMOOTH);    // smooth shading
    glPushMatrix();
    glRotated(0, 1, 0, 0);
    glColor3f(1.0,1.0,1.0);
    
    if (object_3d_z <= radius-20.0)
    {
        switch(rendering_object_list)
        {
            case 0:glutSolidTeapot(4);
                break;
            case 1:glutSolidTorus(3,4,20,20);
                break;
            case 2:glutSolidCone(5,6,20,20);
        }
    }
    glPopMatrix();
    
    glPopMatrix();
    
    glDisable(GL_DEPTH_TEST);
    glPopMatrix();
    glFlush();
}


void gesture_recognition_using_background_subtraction()
{
    cv::Mat gesture_input_image;
    cv::Mat gesture_mid_image;
    cv::Mat gesture_threshold_image;
    
    cvtColor(image, gesture_input_image, CV_RGB2GRAY);
    cv::resize(gesture_input_image,gesture_input_image,objimage_smallsize);
    
    cv::absdiff( background_image,gesture_input_image, gesture_mid_image);
    cv::resize(gesture_mid_image,gesture_output_image,objimage_largesize,cv::INTER_AREA);
    cv::threshold( gesture_output_image, gesture_threshold_image, 20,255,CV_THRESH_BINARY);
    
    cv::flip(gesture_threshold_image, gesture_threshold_image, -1);
    
    cv::Mat under_right_rect;
    cv::Mat under_left_rect;
    
    // selecting the image
    under_left_rect = gesture_threshold_image(cv::Rect(cv::Point(50.0,570.0),cv::Point(200.0,670.0)));
    under_right_rect = gesture_threshold_image(cv::Rect(cv::Point(1080.0,670.0),cv::Point(1230.0,570.0)));
    
    // counting the white pixels
    int white_left_count = countNonZero(under_left_rect);
    int white_right_count = countNonZero(under_right_rect);
    
    cv::Mat gesture_final_image;
    cvtColor(gesture_threshold_image,gesture_final_image,CV_GRAY2RGB);
    
    // drawing rectangles accordingly
    
    if (white_left_count > 1000)
    {
        rendering_object_list = (rendering_object_list-1)%3;
        cv::rectangle(gesture_final_image,cv::Point(50.0,570.0),cv::Point(200.0,670.0),cv::Scalar(0,255,255),-1, 8);
    }
    else
    {
        cv::rectangle(gesture_final_image,cv::Point(50.0,570.0),cv::Point(200.0,670.0),cv::Scalar(0,0,255),-1, 8);
    }
    
    if (white_right_count > 1000)
    {
        rendering_object_list = (rendering_object_list+1)%3;
        cv::rectangle(gesture_final_image,cv::Point(1080.0,670.0),cv::Point(1230.0,570.0),cv::Scalar(0,255,255),-1, 8);
    }
    else
    {
        cv::rectangle(gesture_final_image,cv::Point(1080.0,670.0),cv::Point(1230.0,570.0),cv::Scalar(0,255,0),-1, 8);
    }
    
    
    if(threshold_scene)
    {
        
        glDrawPixels( gesture_final_image.size().width, gesture_final_image.size().height, GL_BGR, GL_UNSIGNED_BYTE, gesture_final_image.ptr() );
        
    }
    
    
}

void gesture_recognition_using_optical_flow()
{
    cv::Mat gesture_input_image ;
    cvtColor(image, gesture_input_image, CV_RGB2GRAY);
    cv::resize(gesture_input_image,gesture_input_image,objimage_smallsize);
    
    cv::Mat computedFlowImage;
    cv::calcOpticalFlowFarneback(background_image,gesture_input_image,computedFlowImage,0.4,1,1,1,5,1.1,cv::OPTFLOW_FARNEBACK_GAUSSIAN);
    
    cv::Mat splitedFlow[2];
    split(computedFlowImage,splitedFlow);
    
    cv::Mat final_gesture_using_optical_flow;
    
    cv::Mat thresholded_gesture_using_optical_flow(cv::Size(320,180), CV_8U);
    for(int i = 0 ; i < 180; i ++)
    {
        for(int j = 0 ; j < 320; j ++)
        {
            //normalisation
            
            double abs_split_0 = abs(splitedFlow[0].at<float>(i,j));
            double abs_split_1 = abs(splitedFlow[1].at<float>(i,j));
            double normalised = sqrt((abs_split_0*abs_split_0)+(abs_split_1*abs_split_1));
            
            //thresholding
            if(normalised > 10)
                thresholded_gesture_using_optical_flow.at<uchar>(i,j) =(uchar)255;
            else
                thresholded_gesture_using_optical_flow.at<uchar>(i,j) = (uchar)0;
        }
    }
    
    cv::resize(thresholded_gesture_using_optical_flow,final_gesture_using_optical_flow,cv::Size(width,height));
    cv::flip(final_gesture_using_optical_flow,final_gesture_using_optical_flow,-1);
    
    
    cv::Mat under_right_rect;
    cv::Mat under_left_rect;
    
    under_left_rect = final_gesture_using_optical_flow(cv::Rect(cv::Point(50.0,570.0),cv::Point(200.0,670.0)));
    under_right_rect = final_gesture_using_optical_flow(cv::Rect(cv::Point(1080.0,670.0),cv::Point(1230.0,570.0)));
    
    // counting the white pixels
    int white_left_count = countNonZero(under_left_rect);
    int white_right_count = countNonZero(under_right_rect);
    
    cv::Mat gesture_final_image;
    cvtColor(final_gesture_using_optical_flow,gesture_final_image,CV_GRAY2BGR);
    if (white_left_count > 2000)
    {
        rendering_object_list = (rendering_object_list-1)%3;
        cv::rectangle(gesture_final_image,cv::Point(50.0,570.0),cv::Point(200.0,670.0),cv::Scalar(0,255,255),-1, 8);
    }
    else
    {
        cv::rectangle(gesture_final_image,cv::Point(50.0,570.0),cv::Point(200.0,670.0),cv::Scalar(0,0,255),-1, 8);
    }
    
    if (white_right_count > 2000)
    {
        rendering_object_list = (rendering_object_list+1)%3;
        cv::rectangle(gesture_final_image,cv::Point(1080.0,670.0),cv::Point(1230.0,570.0),cv::Scalar(0,255,255),-1, 8);
    }
    else
    {
        cv::rectangle(gesture_final_image,cv::Point(1080.0,670.0),cv::Point(1230.0,570.0),cv::Scalar(0,255,0),-1, 8);
    }
    
    if(threshold_scene)
    {
        glDrawPixels( gesture_final_image.size().width, gesture_final_image.size().height, GL_BGR, GL_UNSIGNED_BYTE, gesture_final_image.ptr() );
    }
}


void display()
{
    // clear the window
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // show the current camera frame
    (*cap)>>image;
    
    //Mirroring effect here
    
    cv::flip(image, tempimage, -1);
    glDrawPixels( tempimage.size().width, tempimage.size().height, GL_BGR, GL_UNSIGNED_BYTE, tempimage.ptr() );
    
    //set viewport
    glViewport(0, 0, tempimage.size().width, tempimage.size().height);
    
    //set projection matrix using intrinsic camera params
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(fovy, tempimage.size().width*1.0/tempimage.size().height, 0.01, 600);
    
    // face detection
    cv::Mat tempimage_1;
    cv::flip(image, tempimage_1, -1); // try removing the flip and seeing
    cv:: Mat objimage_gray;
    cv:: Mat objimage_small;
    cvtColor(tempimage_1,objimage_gray,CV_RGB2GRAY);
    cv::resize(objimage_gray,objimage_small,objimage_smallsize);
    cascade.load(cascade_frontal_name);
    cascade.detectMultiScale(objimage_small, rectangles, 1.1, 2,0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(50, 50));
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    if ( rectangles.size()>0)
    {
        cv::transpose(viewMatrix , glViewMatrix);
        glLoadMatrixd(&glViewMatrix.at<double>(0, 0));
    }
    
    
    
    
    // FINAL DRAWING
    
    if (face_border)
    {
        draw_rectangle();
    }
    if (red_circles)
    {
        draw_red_circles();
    }
    if (tea_pot)
    {
        draw_teapot();
    }
    if (gesture_1)
    {
        gesture_recognition_using_background_subtraction();
    }
    if (gesture_2)
    {
        gesture_recognition_using_optical_flow();
    }
    
    
    glutSwapBuffers();
    glutPostRedisplay();
}

void reshape( int w, int h )
{
    // set OpenGL viewport (drawable area)
    glViewport( 0, 0, w, h );
}


void keyboard( unsigned char key, int x, int y )
{
    switch ( key )
    {
        case 'q':
            // quit when q is pressed
            exit(0);
            break;
        case 'r':  // rectangle around face
            if (face_border)
            {face_border= false;
                cout<<"\n Face Border Off" ;}
            else
            {face_border = true;
                cout<<"\n Face Border On" ;
            } ;
            break;
        case 's':  // 3d spheres imposed on rectangle.
            if (red_circles)
            {red_circles= false;
                cout<<"\n Opengl Sphere Off ";}
            else
            {red_circles = true;
                cout<<"\n Opengl Sphere On ";} ;
            break;
        case 't':  // teapot
            if (tea_pot)
            {tea_pot= false;
                cout<<"\n Tea Pot Off ";}
            else
            {tea_pot = true;
                cout<<"\n Tea Pot On ";} ;
            break;
        case 'f':  // gesture recognition using background subtraction
            cvtColor(image,background_image,CV_RGB2GRAY);
            cv::resize(background_image, background_image, objimage_smallsize);
            if (gesture_1)
            {gesture_1= false;
                gesture_2= true;
                cout<<"\n Optical Flow Gesture ";}  // do with previous frame . 
            else
            {gesture_1 = true;
                gesture_2 = false;
                cout<<"\n Background Subtraction Gesture ";} ;
            break;
        case 'g':  // gesture recognition figure using background subtraction
            if (threshold_scene)
            {threshold_scene= false;
                cout<<"\n Threshold Off ";}
            else
            {threshold_scene = true;
                cout<<"\n Threshold On ";} ;
            break;
        default:
            break;
    }
}

void idle()
{
    // grab a frame from the camera
    (*cap) >> image;
    
    // PARAMETERS FOR 2D TO 3D
    if (rectangles.size()>0)
    {
        r1 = rectangles[0];
        
        double face_image_height = r1.height;
        // distance of face from camera
        double Z = fx*face_length/face_image_height;
        face_distance = Z;
        
        cv::Rect scaled_face = r1;
        scaled_face.x*= 4.0;
        scaled_face.y*= 4.0;
        scaled_face.width*= 4.0;
        scaled_face.height*= 4.0;
        
        cv::Point3d point1(scaled_face.x,scaled_face.y,Z);
        cv::Point3d point2(scaled_face.x+scaled_face.width, scaled_face.y,Z );
        cv::Point3d point3(scaled_face.x,scaled_face.y+scaled_face.height,Z );
        cv::Point3d point4(scaled_face.x+scaled_face.width, scaled_face.y+scaled_face.height,Z);
        cv::Point3d center_point5(scaled_face.x+scaled_face.width*0.5,scaled_face.y+scaled_face.height*0.5,Z);
        
        
        // locating 3d x ,y coordinates of the face corners
        face_3d_points[0]= point1;
        face_3d_points[1]= point2;
        face_3d_points[2]= point3;
        face_3d_points[3]= point4;
        face_3d_points[4]= center_point5;
        
        for (int i = 0 ; i <5 ;i++)
        {
            face_3d_points[i].x= (face_3d_points[i].x-(image.size().width/2.0))*Z/fx ;
            face_3d_points[i].y= (face_3d_points[i].y-(image.size().height/2.0))*Z/fy;
        }
        
        objpoint.push_back(face_3d_points[0]);
        objpoint.push_back(face_3d_points[1]);
        objpoint.push_back(face_3d_points[2]);
        objpoint.push_back(face_3d_points[3]);
        objpoint.push_back(face_3d_points[4]);
        
        corners.push_back(cv::Point2f(scaled_face.x,scaled_face.y));
        corners.push_back(cv::Point2f(scaled_face.x+scaled_face.width, scaled_face.y));
        corners.push_back(cv::Point2f(scaled_face.x,scaled_face.y+scaled_face.height));
        corners.push_back(cv::Point2f(scaled_face.x+scaled_face.width, scaled_face.y+scaled_face.height));
        corners.push_back(cv::Point2f(scaled_face.x+scaled_face.width*0.5,scaled_face.y+scaled_face.height*0.5));
        
        cv::Mat rvec;
        cv::Mat tvec;
        cv::Mat rotation;
        cv::solvePnP(objpoint,corners,cameramatrix,distortionCoefficients,rvec,tvec);
        Rodrigues(rvec, rotation);
        
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                viewMatrix.at<double>(row, col) = rotation.at<double>(row, col);
            }
            viewMatrix.at<double>(row, 3) = tvec.at<double>(row, 0);
        }
        viewMatrix.at<double>(3,3)=1.0f;
        
        cv::Mat cvToGl = cv::Mat::zeros(4, 4, CV_64F);
        cvToGl.at<double>(0, 0) = 1.0f;
        cvToGl.at<double>(1, 1) = -1.0f;
        cvToGl.at<double>(2, 2) = -1.0f;
        cvToGl.at<double>(3, 3) = 1.0f;
        viewMatrix = cvToGl * viewMatrix;
        
    }
    
    
    glutPostRedisplay();
    
}

int main( int argc, char **argv )
{
    int w,h;
    
    ifstream calibration_file;
    // start video capture from camera
    cap = new cv::VideoCapture(0);
    calibration_file.open("/Users/akibmayadav/Documents/OpenCV_Fall2016/Magic_Mirror/Magic_Mirror/cam.txt");
    calibration_file >> fovx >> fovy >> fx >>fy>>px>> py >>aspect>>k1>>k2>>p1>>p2>>k3>>focal_length;
    
    if ( argc == 1 ) {
        // start video capture from camera
        cap = new cv::VideoCapture(0);
    } else if ( argc == 2 ) {
        // start video capture from file
        cap = new cv::VideoCapture(argv[1]);
    } else {
        fprintf( stderr, "usage: %s [<filename>]\n", argv[0] );
        return 1;
    }
    
    // check that video is opened
    if ( cap == NULL || !cap->isOpened() ) {
        fprintf( stderr, "could not start video capture\n" );
        return 1;
    }
    
    cout<<"\n fovx"<<fovx<<"\n fovy"<<fovy<<"\n fx:"<<fx<<"\n fy:"<<fy<<"\n px:"<<px<<"\n py:"<<py<<"\n aspect:"<<aspect<<"\n k1:"<<k1<<"\n k2:"<<k2<<"\n k3:"<<k3<<"\n p1:"<<p1<<"\n p2:"<<p2;
    
    // get width and height
    w = (int) cap->get( CV_CAP_PROP_FRAME_WIDTH );
    h = (int) cap->get( CV_CAP_PROP_FRAME_HEIGHT );
    // On Linux, there is currently a bug in OpenCV that returns
    // zero for both width and height here (at least for video from file)
    // hence the following override to global variable defaults:
    width = w ? w : width;
    height = h ? h : height;
    
    
    //DISTORTION MATRIX
    
    double dist[] = {k1,k2,p1,p2,k3};
    distortionCoefficients=cv::Mat(5,1,CV_64FC1,dist);
    cout<<"\n dist coeff:"<<distortionCoefficients;
    
    //CAMERA MATRIX
    double camera[] = {fx,0.,width/2.0,
        0.,fy,height/2.0,
        0.,0.,1.};
    cameramatrix=cv::Mat(3,3,CV_64FC1,camera);
    cout<<"\n camera matrix:"<<cameramatrix<<endl;
    
    
    // initialize GLUT
    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE );
    glutInitWindowPosition( 20, 20 );
    glutInitWindowSize( width, height );
    glutCreateWindow( "OpenGL / OpenCV Example" );
    
    // set up GUI callback functions
    glutDisplayFunc( display );
    glutReshapeFunc( reshape );
    glutKeyboardFunc( keyboard );
    glutIdleFunc( idle );
    // start GUI loop
    glutMainLoop();
    
    return 0;
}
