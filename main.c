#include <GL/freeglut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "mlp.h"
#include "problem.h"

#define HIST_SIZE 512

#ifndef AIGSIM_PATH
// refer to https://github.com/arminbiere/aiger for setup
#define AIGSIM_PATH "aigsim"
#endif

#define LOG_DEBUG(fmt, ...) fprintf(stdout, "[DEBUG] " fmt "\n", ##__VA_ARGS__)

#define LOG_INFO(fmt, ...) fprintf(stdout, "[INFO ] " fmt "\n", ##__VA_ARGS__)

#define LOG_WARN(fmt, ...) fprintf(stdout, "[WARN ] " fmt "\n", ##__VA_ARGS__)

#define LOG_ERROR(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)

#define LOG_SUCCESS(fmt, ...) fprintf(stdout, "[ OK   ] " fmt "\n", ##__VA_ARGS__)

typedef struct {
  double lr;
  int paused;
  int steps_per_frame;
} TrainCtl;

static MLP net;
static TrainCtl ctl = {INIT_LR, 0, INIT_STEPS_PER_FRAME};
static double current_loss = 0.0;
static double current_accuracy = -1.0;
static int show_net = 0;

static double loss_hist[HIST_SIZE];
static int hist_head = 0;
static int hist_count = 0;

static int win_w = 1000;
static int win_h = 700;

static void push_loss(double v) {
  loss_hist[hist_head] = v;
  hist_head = (hist_head + 1) % HIST_SIZE;
  if (hist_count < HIST_SIZE) hist_count++;
}

static double hist_get(int k) {
  int idx = (hist_head - hist_count + k) % HIST_SIZE;
  if (idx < 0) idx += HIST_SIZE;
  return loss_hist[idx];
}

static double train_one_epoch(void) {
  bool inputs[INPUT_SIZE];
  double x[INPUT_SIZE];
  bool outputs[OUTPUT_SIZE];
  double t[OUTPUT_SIZE];

  double loss = 0.0;
  for (int s = 0; s < ctl.steps_per_frame; ++s) {
    dataset_make_train_sample(inputs, outputs);
    for (int i = 0; i < INPUT_SIZE; ++i) x[i] = inputs[i] ? 1.0 : -1.0;
    for (int i = 0; i < OUTPUT_SIZE; ++i) t[i] = outputs[i] ? 1.0 : -1.0;
    mlp_forward(&net, x);
    loss += mlp_backward(&net, t, ctl.lr);
  }
  return loss / (double)ctl.steps_per_frame;
}

static double test_accuracy(int samples) {
  bool inputs[INPUT_SIZE];
  double x[INPUT_SIZE];
  bool outputs[OUTPUT_SIZE];

  int correct = 0;

  for (int s = 0; s < samples; ++s) {
    test_validator_t validator;

    dataset_make_test_sample(inputs, &validator);

    for (int i = 0; i < INPUT_SIZE; ++i) {
      x[i] = inputs[i] ? 1.0 : -1.0;
    }

    mlp_forward(&net, x);

    for (int i = 0; i < OUTPUT_SIZE; ++i) {
      outputs[i] = net.a[net.nb_layers - 1][i] > 0.0;
    }

    correct += dataset_validate_output(&validator, outputs);
  }

  return (double)correct / (double)samples;
}

static double test_aiger_accuracy(int samples, const char* path) {
  char line[4096];

  LOG_INFO("Testing (%d samples, file: %s)...", samples, path);
  LOG_DEBUG("Pipes initialisation...");

  int pipe_in[2];
  int pipe_out[2];

  if (pipe(pipe_in) == -1 || pipe(pipe_out) == -1) {
    LOG_ERROR("fail on pipe()");
    return -1.;
  }

  LOG_DEBUG("Trying to fork()...");
  pid_t pid = fork();
  if (pid == -1) {
    LOG_ERROR("fail on fork()");
    return -1.;
  }

  if (pid == 0) {
    // --- PROCESSUS FILS (aigsim) ---
    dup2(pipe_in[0], STDIN_FILENO);
    dup2(pipe_out[1], STDOUT_FILENO);

    close(pipe_in[0]);
    close(pipe_in[1]);
    close(pipe_out[0]);
    close(pipe_out[1]);

    // execl(cmd_path, cmd_path, path, (char*)NULL);
    execl("/usr/bin/stdbuf", "stdbuf", "-oL", AIGSIM_PATH, path, (char*)NULL);
    LOG_ERROR("execl failed starting aigsim");
    exit(EXIT_FAILURE);
  }

  // --- PROCESSUS PARENT ---
  close(pipe_in[0]);
  close(pipe_out[1]);

  FILE* to_pipe = fdopen(pipe_in[1], "w");
  FILE* from_pipe = fdopen(pipe_out[0], "r");

  if (!to_pipe || !from_pipe) {
    LOG_ERROR("failure fdopen()");
    if (to_pipe)
      fclose(to_pipe);
    else
      close(pipe_in[1]);
    if (from_pipe)
      fclose(from_pipe);
    else
      close(pipe_out[0]);
    return -1.;
  }

  LOG_INFO("Process aigsim started with PID: %d\n", pid);

  bool inputs[INPUT_SIZE];
  bool outputs[OUTPUT_SIZE];
  int correct = 0;

  for (int s = 0; s < samples; ++s) {
    test_validator_t validator;
    dataset_make_test_sample(inputs, &validator);

    // Préparation de la chaîne de debug pour l'envoi
    LOG_INFO(" %d/%d -> Sending: ", s + 1, samples);
    for (int i = 0; i < INPUT_SIZE; ++i) {
      char c = inputs[i] ? '1' : '0';
      putchar(c);  // Debug sur la console
      fputc(c, to_pipe);
    }
    fputc('\n', to_pipe);
    putchar('\n');

    LOG_DEBUG("fflush(to_pipe) sent. waiting for aigsim...");
    fflush(to_pipe);

    if (!fgets(line, sizeof(line), from_pipe)) {
      LOG_ERROR("aigsim stoped the pipe");
      fclose(to_pipe);
      fclose(from_pipe);
      waitpid(pid, NULL, 0);
      return -1.;
    }

    LOG_INFO("<- Received: '%s'", line);  // line contient déjà le \n

    for (int i = 0; i < OUTPUT_SIZE; i++) {
      outputs[i] = line[i + INPUT_SIZE + 2] == '1';
    }

    int res = dataset_validate_output(&validator, outputs);
    correct += res;
    LOG_INFO("Validation: %s\n", res ? "CORRECT" : "FALSE");
  }

  LOG_INFO("Closing pipes...");
  fclose(to_pipe);
  fclose(from_pipe);

  LOG_INFO("Waiting child process (waitpid)...");
  int status;
  waitpid(pid, &status, 0);
  LOG_DEBUG("Child Process ended with: %d\n", status);

  return (double)correct / (double)samples;
}

static void draw_text(float x, float y, const char* s) {
  glRasterPos2f(x, y);
  for (const char* p = s; *p; ++p) {
    glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *p);
  }
}

static void draw_hud(void) {
  char buf[256];

  snprintf(buf, sizeof(buf),
           "pause=%s  lr=%.6f  steps/frame=%d  Wtemp=%.3f  Otemp=%.3f  loss=%.8f accuracy=%.2f%%",
           ctl.paused ? "ON" : "OFF", ctl.lr, ctl.steps_per_frame, net.weights_temp, net.out_temp,
           current_loss, current_accuracy * 100.f);
  draw_text(20.0f, win_h - 25.0f, buf);

  draw_text(20.0f, win_h - 45.0f,
            "p pause | w net-wiew | g export | n step | +/- lr | [ / ] steps | 1/2 Wtemp | 3/4 Otemp | r reset "
            "| ESC quit");
}

static void draw_graph(void) {
  const float x0 = 40.0f;
  const float y0 = 80.0f;
  const float w = (float)win_w - 80.0f;
  const float h = (float)win_h - 170.0f;

  if (hist_count < 2) return;

  double lo = hist_get(0), hi = hist_get(0);
  for (int i = 1; i < hist_count; ++i) {
    double v = hist_get(i);
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (fabs(hi - lo) < 1e-12) {
    hi = lo + 1.0;
  }

  glBegin(GL_LINE_STRIP);
  for (int i = 0; i < hist_count; ++i) {
    double v = hist_get(i);
    float x = x0 + w * (float)i / (float)(HIST_SIZE - 1);
    float y = y0 + h * (float)((v - lo) / (hi - lo));
    glVertex2f(x, y);
  }
  glEnd();

  glBegin(GL_LINE_LOOP);
  glVertex2f(x0, y0);
  glVertex2f(x0 + w, y0);
  glVertex2f(x0 + w, y0 + h);
  glVertex2f(x0, y0 + h);
  glEnd();
}

static void draw_selected_connections(void) {
  glBegin(GL_LINES);

  for (unsigned l = 1; l < net.nb_layers; ++l) {
    float x2 = 80.0f + (float)l * ((win_w - 160.0f) / (net.nb_layers - 1));

    float x1 = 80.0f + (float)(l - 1) * ((win_w - 160.0f) / (net.nb_layers - 1));

    for (unsigned j = 0; j < net.sizes[l]; ++j) {
      float y2 = 80.0f + (float)(j + 1) * ((win_h - 200.0f) / (net.sizes[l] + 1));

      for (int k = 0; k < NB_ENTRY; ++k) {
        int i = net.I[l][j][k];

        if ((unsigned)i >= (unsigned)net.sizes[l - 1]) continue;

        float y1 = 80.0f + (float)(i + 1) * ((win_h - 200.0f) / (net.sizes[l - 1] + 1));

        double w = tanh(net.W[l][j][k] / net.weights_temp);

        float mag = fminf(fabsf((float)w), 1.0f);

        // bleu = positif
        // rouge = négatif
        glColor3f(w < 0 ? mag : 0.0f, 0.0f, w > 0 ? mag : 0.0f);

        glVertex2f(x1, y1);
        glVertex2f(x2, y2);
      }
    }
  }

  glEnd();

  // neurones
  for (unsigned l = 1; l < net.nb_layers; ++l) {
    float x = 80.0f + (float)l * ((win_w - 160.0f) / (net.nb_layers - 1));

    for (unsigned j = 0; j < net.sizes[l]; ++j) {
      float y = 80.0f + (float)(j + 1) * ((win_h - 200.0f) / (net.sizes[l] + 1));

      float pos = fabs(net.b[l][j]) > 1.0f ? 1.f : 0.f;

      float mag = fminf(fabsf((float)net.b[l][j]), 1.0f);

      // bleu = positif
      // rouge = négatif
      glColor3f(net.b[l][j] < 0 ? mag : 0.0f, pos, net.b[l][j] > 0 ? mag : 0.0f);

      float s = 4.0f;
      glBegin(GL_QUADS);
      glVertex2f(x - s, y - s);
      glVertex2f(x + s, y - s);
      glVertex2f(x + s, y + s);
      glVertex2f(x - s, y + s);
      glEnd();
    }
  }

  glColor3f(1.0f, 1.0f, 1.0f);
}

static void display(void) {
  glClear(GL_COLOR_BUFFER_BIT);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  draw_graph();
  if (show_net) draw_selected_connections();
  draw_hud();

  glutSwapBuffers();
}

static void reshape(int w, int h) {
  win_w = (w > 1) ? w : 1;
  win_h = (h > 1) ? h : 1;

  glViewport(0, 0, win_w, win_h);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0.0, (double)win_w, 0.0, (double)win_h);
}

static void reset_history(void) {
  hist_head = 0;
  hist_count = 0;
  memset(loss_hist, 0, sizeof(loss_hist));
  current_loss = 0.0;
}

static void reset_net(void) {
  mlp_randomize(&net);
  reset_history();
}

static void keyboard(unsigned char key, int x, int y) {
  (void)x;
  (void)y;

  switch (key) {
    case 27:
    case 'q':
      mlp_free(&net);
      exit(0);
      break;

    case 'p':
    case 'P':
      ctl.paused = !ctl.paused;
      break;

    case 'n':
      if (ctl.paused) {
        current_loss = train_one_epoch();
        push_loss(current_loss);
      }
      break;

    case '+':
    case '=':
      ctl.lr *= 1.1;
      break;

    case '-':
    case '_':
      ctl.lr *= 0.9;
      break;

    case '[':
      if (ctl.steps_per_frame > 1) ctl.steps_per_frame--;
      break;

    case ']':
      ctl.steps_per_frame++;
      break;

    case '1':
      mlp_set_temps(&net, net.weights_temp * 0.9, net.out_temp);
      break;

    case '2':
      mlp_set_temps(&net, net.weights_temp * 1.1, net.out_temp);
      break;

    case '3':
      mlp_set_temps(&net, net.weights_temp, net.out_temp * 0.9);
      break;

    case '4':
      mlp_set_temps(&net, net.weights_temp, net.out_temp * 1.1);
      break;

    case 'r':
    case 'R':
      reset_net();
      break;

    case 's':
      LOG_INFO("Saving MLP in './out/mlp.bin'...");
      mlp_save(&net, "./out/mlp.bin");
      LOG_SUCCESS("Done !");
      break;
    case 'l':
      LOG_INFO("Loading MLP from './out/mlp.bin'...");
      mlp_free(&net);
      net = mlp_load("./out/mlp.bin");
      LOG_SUCCESS("Done !");
      break;

    case 'w':
    case 'W':
      show_net = !show_net;
      break;

    case 'a':
    case 'A':
      LOG_INFO("Testing accuracy...\n");

      current_accuracy = test_accuracy(1000);

      LOG_SUCCESS("Accuracy: %.2f%%\n", current_accuracy * 100.0);
      break;

    case 'g':
    case 'G':
      LOG_INFO("Exporting to './out/out.aig'...\n");
      FILE* f = fopen("./out/out.aig", "w");
      if (!f) break;
      mlp_aiger_export(&net, aiger_binary_mode, f);
      fclose(f);
      LOG_SUCCESS("'./out/out.aig' exported !");

      current_accuracy = test_aiger_accuracy(1000, "./out/out.aig");

      LOG_SUCCESS("Gate net accuracy : %.1f%% !", current_accuracy*100.f);
      break;
  }

  glutPostRedisplay();
}

static void idle(void) {
  if (!ctl.paused) {
    double loss = train_one_epoch();
    current_loss = loss;
    push_loss(current_loss);
  }

  glutPostRedisplay();
}

static void cleanup(void) {
  LOG_INFO("Cleaning up...\n");
  dataset_destroy();
  mlp_free(&net);
}

static void init(void) {
  LOG_INFO("Initialising dataset...");
  dataset_init();
  LOG_SUCCESS("Dataset Initialised !");

  LOG_INFO("Initialising MLP...");
  srand((unsigned)time(NULL));
  net = mlp_create(nb_layers, layers, INIT_WTEMP, INIT_OTEMP);
  if (!net.meta_block || !net.data_block) {
    LOG_ERROR("Memory allocation failed");
  }
  LOG_SUCCESS("MLP Initialised !");
}

int main(int argc, char** argv) {
  init();

  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
  glutInitWindowSize(win_w, win_h);
  glutCreateWindow("MLP live loss");

  glClearColor(0.08f, 0.08f, 0.09f, 1.0f);
  glColor3f(1.0f, 1.0f, 1.0f);

  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutKeyboardFunc(keyboard);
  glutCloseFunc(cleanup);
  glutIdleFunc(idle);

  glutMainLoop();

  return 0;
}