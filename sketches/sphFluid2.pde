import queasycam.*;

String projectTitle = "SPH Fluid Sim";

QueasyCam cam;

float floor = 695;
float ceiling = 0;
float left = 0;
float right = 1280;
float delta = 15;
float dtime = .01;
float gravity = 4;
float pull = .2;

float pointsize = 15;

float smooth = 35;
float stiff = 100000;
float stiffN = 2500;
float rest = .2;

int numpoints = 1000;

float grablength = 100;

float pressred = 5000*25;
float pressblue = 200*25;

boolean upgrav = false;
boolean downgrav = false;
boolean leftgrav = false;
boolean rightgrav = false;

float boundary_stride = smooth/16;
float boundary_height = smooth;

//float m_b = 4*rest*smooth*smooth/4;
float m_p = rest*right*300.0/numpoints;

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
  ArrayList<Particle> boundary_particles;
  
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
    boundary_particles = new ArrayList<Particle>();
    
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
    /*
      1 boundary particle for a width of x-smooth/4 to x+smooth/4 and a height of y-smooth/4 tp y+smooth/4
      thus a smooth/2 by smooth/2 rectangle -> V = smooth^2/4
                                            -> rest = m_b / V
                                            -> m_b = rest*V = rest*smooth*smooth/4
    */
    for(float i=0; i<right; i+=boundary_stride){
      boundary_particles.add(new Particle(i, floor));
      /*boundary_particles.add(new Particle(i, floor+boundary_stride));
      boundary_particles.add(new Particle(i, floor+2*boundary_stride));
      boundary_particles.add(new Particle(i, floor+3*boundary_stride));*/
    }
    for(float i=0; i<700; i+=boundary_stride){
       boundary_particles.add(new Particle(left, floor-i));
       /*boundary_particles.add(new Particle(left-boundary_stride, floor-i));
       boundary_particles.add(new Particle(left-2*boundary_stride, floor-i));
       boundary_particles.add(new Particle(left-3*boundary_stride, floor-i));*/
    }
    for(float i=0; i<700; i+=boundary_stride){
       boundary_particles.add(new Particle(right, floor-i));
       /*boundary_particles.add(new Particle(right+boundary_stride, floor-i));
       boundary_particles.add(new Particle(right+2*boundary_stride, floor-i));
       boundary_particles.add(new Particle(right+3*boundary_stride, floor-i));*/
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
      float max_dens = 0;
      ArrayList<Pair> pairs = new ArrayList<Pair>();
      ArrayList<Pair> boundary_pairs = new ArrayList<Pair>();
    
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
        if (p.x < left || p.x > right) {
          if (p.x < left) {
            p.x = left+boundary_height/2;
          } else {
            p.x = right-boundary_height/2;
          }
          p.velx *= 0.0;
        }
        if (p.y > floor) {
          p.y = floor-boundary_height/2;
          p.vely *= 0.0;
        }
        
        
        p.oldx = p.x;
        p.oldy = p.y;
        p.x += p.velx * dt;
        p.y += p.vely * dt;
        p.dens = 0;
        p.densN = 0;
        p.velx = 0;
        p.vely = 0;
      }
      
      for (int i = 0; i < num; i++) {
        for (int j = 0; j < i; j++) {
          Particle p1 = particles.get(i);
          Particle p2 = particles.get(j);
          float dist = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
          if (dist < ksmooth) {
            pairs.add(new Pair(p1, p2));
          }
        }
      }
      
      for (int i = 0; i < num; i++) {
        for (int j = 0; j < boundary_particles.size(); j++) {
          Particle p1 = particles.get(i);
          Particle p2 = boundary_particles.get(j);
          p2.dens = 0;
          p2.densN = 1000;
          float dist = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
          if (dist < ksmooth) {
            boundary_pairs.add(new Pair(p1, p2));
          }
        }
      }
      
      for (int i = 0; i < pairs.size(); i++) {
        Pair p = pairs.get(i);
        float dist = sqrt(pow(p.a.x - p.b.x, 2) + pow(p.a.y - p.b.y, 2));
        p.q = (float)(2 * Math.exp( -dist*dist / (smooth*smooth/4)) / Math.pow(smooth/2, 4) / Math.PI);
        //p.q = (float)2*(1 - dist / ksmooth)/641.409;
        p.q2 = (float)(Math.pow(1.0 / ((smooth/2)*Math.sqrt(Math.PI)), 2) * Math.exp( -dist*dist / (smooth*smooth/4)));
        //p.q2 = pow(1 - dist / ksmooth, 2)/641.409;
        p.a.dens += m_p*p.q2;
        p.b.dens += m_p*p.q2;
      }
      
      for (int i = 0; i < boundary_pairs.size(); i++) {
        Pair p = boundary_pairs.get(i);
        float multiplier = 3.0;
        if(p.b.dens < multiplier*p.a.dens){
          p.b.dens = multiplier*p.a.dens;
        }
      }

      for (int i = 0; i < boundary_pairs.size(); i++) {
        Pair p = boundary_pairs.get(i);
        float dist = sqrt(pow(p.a.x - p.b.x, 2) + pow(p.a.y - p.b.y, 2));
        //p.q = 2*(1 - dist / ksmooth)/641.409;
        p.q = (float)(2 * Math.exp( -dist*dist / (smooth*smooth/4)) / Math.pow(smooth/2, 4) / Math.PI);
        //p.q2 = pow(1 - dist / ksmooth, 2)/641.409;
        p.q2 = (float)(Math.pow(1.0 / ((smooth/2)*Math.sqrt(Math.PI)), 2) * Math.exp( -dist*dist / (smooth*smooth/4)));
        p.a.dens += (boundary_stride*boundary_height*p.b.dens)*p.q2;
      }
      
      for (int i = 0; i < num; i++) {
        Particle p = particles.get(i);
        p.press = kstiff * (p.dens - krest);
      }
      
      for (int i = 0; i < pairs.size(); i++) {
        Pair p = pairs.get(i);
        float press = m_p*(p.a.press/(p.a.dens*p.a.dens) + p.b.press/(p.b.dens*p.b.dens));
        float displace = (press * p.q) * dt;
        //float dist = sqrt(pow(p.a.x - p.b.x, 2) + pow(p.a.y - p.b.y, 2));
        float abx = (p.a.x - p.b.x);
        float aby = (p.a.y - p.b.y);
        p.a.velx += displace * abx;
        p.a.vely += displace * aby;
        p.b.velx -= displace * abx;
        p.b.vely -= displace * aby;
      }
      
      for (int i = 0; i < boundary_pairs.size(); i++) {
        Pair p = boundary_pairs.get(i);
        float press = (boundary_stride*boundary_height*p.b.dens)*(p.a.press/(p.a.dens*p.a.dens));
        float displace = (press * p.q) * dt;
        //float dist = sqrt(pow(p.a.x - p.b.x, 2) + pow(p.a.y - p.b.y, 2));
        float abx = (p.a.x - p.b.x);
        float aby = (p.a.y - p.b.y);
        p.a.velx += displace * abx;
        p.a.vely += displace * aby;
      }
      
      for (int i = 0; i < num; i++) {
        Particle p = particles.get(i);
        p.x += p.velx * dt;
        p.y += p.vely * dt;
      }
      surface.setTitle("Test "+max_dens+" "+rest);
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
  
  /*String runtimeReport = "Frame: "+str(endFrame-startFrame)+"ms,"+
        " Physics: "+ str(endPhysics-startFrame)+"ms,"+
        " FPS: "+ str(round(frameRate)) + "\n";
  surface.setTitle(projectTitle+ "  -  " +runtimeReport);*/
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
