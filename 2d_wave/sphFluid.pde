import queasycam.*;

String projectTitle = "SPH Fluid Sim";

QueasyCam cam;

float floor = 695;
float ceiling = 0;
float left = 0;
float right = 1280;
float delta = 15;
float dtime = .01;
float gravity = 9.8;
float pull = .2;

float pointsize = 15;

float smooth = 35;
float stiff = 1500;
float stiffN = 2500;
float rest = .2;

int numpoints = 1000;

float grablength = 100;

float pressred = 5000;
float pressblue = 200;

boolean upgrav = false;
boolean downgrav = false;
boolean leftgrav = false;
boolean rightgrav = false;

class Vector {
  float x, y, z;
  
  Vector(float a, float b, float c) {
    x = a;
    y = b;
    z = c;
  }
  
  void set(Vector v) {
    x = v.x;
    y = v.y;
    z = v.z;
  }
  
  void vnormalize() {
    float length = sqrt(pow(x,2) + pow(y, 2) + pow(z, 2));
    x /= length;
    y /= length;
    z /= length;
  }
}

Vector vadd(Vector a, Vector b) {
  return new Vector(a.x + b.x, a.y + b.y, a.z + b.z);
}

Vector vsubtract(Vector a, Vector b) {
  return new Vector(a.x - b.x, a.y - b.y, a.z - b.z);
}

Vector vmult(float a, Vector v) {
  return new Vector(a * v.x, a * v.y, a * v.z);
}

float vdot(Vector a, Vector b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vector vcross(Vector a, Vector b) {
  return new Vector(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

float vdistance(Vector a, Vector b) {
  return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.z - b.z, 2));
}


class Particle {
  float x, y;
  float oldx, oldy;
  float velx, vely;
  float dens, densN;
  float press, pressN;
  boolean grabbed;
  
  Particle(float x, float y) {
    this.x = x;
    this.y = y;
    oldx = x;
    oldy = y;
    velx = 0;
    vely = 0;
    dens = 0;
    densN = 0;
    press = 0;
    pressN = 0;
    grabbed = false;
  }
  
  void drawParticle() {
    float red = 50;
    float green = 50;
    float blue = 255;
    float diff = pressred - pressblue;
    float quad = diff / 4;
    if (press < pressblue) {
    } else if (press < pressblue + quad) {
      float segdiff = press - pressblue;
      green = segdiff / quad * 255;
    } else if (press < pressblue + 2 * quad) {
      float segdiff = press - pressblue - quad;
      green = 255;
      blue = 255 - (segdiff / quad * 255);
    } else if (press < pressblue + 3 * quad) {
      float segdiff = press - pressblue - 2 * quad;
      green = 255;
      blue = 50;
      red = segdiff / quad * 255;
    } else if (press < pressred) {
      float segdiff = press - pressblue - 3 * quad;
      red = 255;
      blue = 50;
      green = 255 - (segdiff / quad * 255);
    } else {
      red = 255;
      blue = 50;
    }
    stroke(red, green, blue);
    point(x, y);
  }
}



class Pair {
  Particle a, b;
  float q, q2, q3;
  
  Pair(Particle x, Particle y) {
    a = x;
    b = y;
    q = q2 = q3 = 0;
  }
}



class SphFluid {
  ArrayList<Particle> particles;
  int num;
  float ksmooth, kstiff, kstiffN, krest;
  float reach;
  
  SphFluid(int num, float ksm, float kst, float kstn, float kr, float reach) {
    this.num = num;
    ksmooth = ksm;
    kstiff = kst;
    kstiffN = kstn;
    krest = kr;
    this.reach = reach;
    particles = new ArrayList<Particle>();
    
    float interval = pointsize * 1.5;
    float initx = interval;
    float inity = interval;
    float row = 0;
    for (int i = 0; i < num; i++) {
      particles.add(new Particle(initx + row * 5, inity));
      initx += interval;
      if (initx > right - interval) {
        initx = interval;
        inity += interval;
        row++;
      }
    }
  }
  
  
  void grab() {
    for (int i = 0; i < num; i++) {
      Particle p = particles.get(i);
      float dist = sqrt(pow(mouseX - p.x, 2) + pow(mouseY - p.y, 2));
      if (dist < reach) {
        p.grabbed = true;
      }
    }
  }
  
  void letGo() {
    for (int i = 0; i < num; i++) {
      Particle p = particles.get(i);
      p.grabbed = false;
    }
  }
  
  
  void updateParticles(float dt) {
    for (int z = 0; z < 2; z++) {
      ArrayList<Pair> pairs = new ArrayList<Pair>();
    
      for (int i = 0; i < num; i++) {
        Particle p = particles.get(i);
        p.velx = (p.x - p.oldx) / dt;
        p.vely = (p.y - p.oldy) / dt;
        
        // gravity
        if (upgrav || leftgrav || rightgrav) {
          if (upgrav) {
            p.vely -= gravity;
          }
          if (leftgrav) {
            p.velx -= gravity;
          }
          if (rightgrav) {
            p.velx += gravity;
          }
          if (downgrav) {
            p.vely += gravity;
          }
        } else {
          p.vely += gravity;
        }
        
        // grab
        if (mousePressed && p.grabbed) {
          float vx = (mouseX - p.x);
          float vy = (mouseY - p.y);
          p.velx += pull * vx;
          p.vely += pull * vy;
        }
        
        // bounds
        if (p.x <= left + pointsize / 2 || p.x >= right - pointsize / 2) {
          if (p.x <= left + pointsize / 2) {
            p.x = left + pointsize / 2 * 1.05;
          } else {
            p.x = right - pointsize / 2 * 1.05;
          }
          p.velx *= -.3;
        }
        if (p.y >= floor - pointsize / 2 || p.y <= ceiling + pointsize / 2) {
          if (p.y >= floor - pointsize / 2) {
            p.y = floor - pointsize / 2 * 1.05;
          } else {
            p.y = ceiling + pointsize / 2 * 1.05;
          }
          p.vely *= -.2;
        }
        
        
        p.oldx = p.x;
        p.oldy = p.y;
        p.x += p.velx * dt;
        p.y += p.vely * dt;
        p.dens = 0;
        p.densN = 0;
      }
      
      for (int i = 0; i < num; i++) {
        for (int j = 0; j < num; j++) {
          if (i != j) {
            Particle p1 = particles.get(i);
            Particle p2 = particles.get(j);
            float dist = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
            if (dist < ksmooth) {
              pairs.add(new Pair(p1, p2));
            }
          }
        }
      }
      
      for (int i = 0; i < pairs.size(); i++) {
        Pair p = pairs.get(i);
        float dist = sqrt(pow(p.a.x - p.b.x, 2) + pow(p.a.y - p.b.y, 2));
        p.q = 1 - dist / ksmooth;
        p.q2 = pow(p.q, 2);
        p.q3 = pow(p.q, 3);
        p.a.dens += p.q2;
        p.b.dens += p.q2;
        p.a.densN += p.q3;
        p.b.densN += p.q3;
      }
      
      for (int i = 0; i < num; i++) {
        Particle p = particles.get(i);
        p.press = kstiff * (p.dens - krest);
        p.pressN = kstiffN * p.densN;
        if (p.press > 6000) {
          p.press = 6000;
        }
        if (p.pressN > 7000) {
          p.pressN = 7000;
        }
      }
      
      for (int i = 0; i < pairs.size(); i++) {
        Pair p = pairs.get(i);
        float press = p.a.press + p.b.press;
        float pressN = p.a.pressN + p.b.pressN;
        float displace = (press * p.q + pressN * p.q2) * pow(dt, 2);
        float dist = sqrt(pow(p.a.x - p.b.x, 2) + pow(p.a.y - p.b.y, 2));
        float abx = (p.a.x - p.b.x) / dist;
        float aby = (p.a.y - p.b.y) / dist;
        p.a.x += displace * abx;
        p.a.y += displace * aby;
        p.b.x -= displace * abx;
        p.b.y -= displace * aby;
      }
    }
  }
  
  void drawParticles() {
    strokeWeight(pointsize);
    for (int i = 0; i < num; i++) {
      Particle p = particles.get(i);
      p.drawParticle();
    }
    noStroke();
  }
}




SphFluid fluid = new SphFluid(numpoints, smooth, stiff, stiffN, rest, grablength);



void setup() {
 size(1280, 695, P3D);
 //cam = new QueasyCam(this);
 //cam.speed = 3;
 //cam.sensitivity = 1;
 noStroke();
}

void computePhysics(float dt) {
  fluid.updateParticles(dt);
}

void drawScene(){
  background(50, 51, 54);
  lights();

  fluid.drawParticles();
}

void draw() {
  float startFrame = millis(); 
  computePhysics(dtime); 
  float endPhysics = millis();
  
  drawScene();
  float endFrame = millis();
  delta = endFrame - startFrame;
  
  String runtimeReport = "Frame: "+str(endFrame-startFrame)+"ms,"+
        " Physics: "+ str(endPhysics-startFrame)+"ms,"+
        " FPS: "+ str(round(frameRate)) + "\n";
  surface.setTitle(projectTitle+ "  -  " +runtimeReport);
}

void keyPressed() {
  if (key == 32) {
    //swater = new ShallowWater(   50,    50,       10);
  }
  
  if (key == 'w') {
    upgrav = true;
  }
  if (key == 'a') {
    leftgrav = true;
  }
  if (key == 'd') {
    rightgrav = true;
  }
  if (key == 's') {
    downgrav = true;
  }
}

void keyReleased() {
  if (key == 'w') {
    upgrav = false;
  }
  if (key == 'a') {
    leftgrav = false;
  }
  if (key == 'd') {
    rightgrav = false;
  }
  if (key == 's') {
    downgrav = false;
  }
}

void mousePressed() {
  fluid.grab();
}

void mouseReleased() {
  fluid.letGo();
}
