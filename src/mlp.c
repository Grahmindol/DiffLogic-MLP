#include "mlp.h"

#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qm16.h"

static double rand_uniform(double a, double b) {
  return a + (b - a) * ((double)rand() / (double)RAND_MAX);
}

static double weight_from_logit(const MLP* net, double l) { return tanh(l / net->weights_temp); }

static double dweight_from_logit(const MLP* net, double l) {
  double t = tanh(l / net->weights_temp);
  return (1.0 - t * t) / net->weights_temp;
}

static double output_from_sum(const MLP* net, double z) { return tanh(z / net->out_temp); }

static double dtanh_from_output(const MLP* net, double y) {
  double v = (1.0 - y * y) / net->out_temp;
  return (v < 1e-8) ? 1e-8 : v;
}

void mlp_set_temps(MLP* net, double weights_temp, double out_temp) {
  if (weights_temp < 0.05) weights_temp = 0.05;
  if (out_temp < 0.05) out_temp = 0.05;
  if (weights_temp > 100.0) weights_temp = 100.0;
  if (out_temp > 100.0) out_temp = 100.0;

  net->weights_temp = weights_temp;
  net->out_temp = out_temp;
}

static void mlp_fill_random(MLP* net) {
  for (unsigned l = 1; l < net->nb_layers; ++l) {
    int nprev = net->sizes[l - 1];

    for (unsigned j = 0; j < net->sizes[l]; ++j) {
      net->b[l][j] = rand_uniform(-0.5, 0.5);

      for (int k = 0; k < NB_ENTRY; ++k) {
        net->W[l][j][k] = rand_uniform(-0.1, 0.1);
        net->I[l][j][k] = (nprev > 0) ? (rand() % nprev) : 0;
      }
    }
  }
}

MLP mlp_create(size_t nb_layers, const size_t* sizes, double weights_temp, double out_temp) {
  MLP net = {0};

  if (nb_layers < 1 || !sizes) return net;

  net.nb_layers = nb_layers;
  net.weights_temp = (weights_temp < 0.05) ? 0.05 : weights_temp;
  net.out_temp = (out_temp < 0.05) ? 0.05 : out_temp;

  // 1. Calculate total elements needed for bulk allocations
  size_t act_count = 0;     // Total activations across all layers
  size_t bias_count = 0;    // Total biases (layers 1..nb_layers-1)
  size_t weight_count = 0;  // Total weights (layers 1..nb_layers-1)
  size_t row_count = 0;     // Total matrix rows (layers 1..nb_layers-1)

  for (size_t l = 0; l < nb_layers; ++l) {
    act_count += sizes[l];
    if (l > 0) {
      bias_count += sizes[l];
      weight_count += sizes[l] * (size_t)NB_ENTRY;
      row_count += sizes[l];
    }
  }

  // 2. Allocate layer metadata pointers (1 allocation per array)
  net.sizes = calloc(nb_layers, sizeof(size_t));
  net.a = calloc(nb_layers, sizeof(double*));
  net.z = calloc(nb_layers, sizeof(double*));
  net.delta = calloc(nb_layers, sizeof(double*));
  net.b = calloc(nb_layers, sizeof(double*));
  net.W = calloc(nb_layers, sizeof(double***));
  net.M = calloc(nb_layers, sizeof(double***));
  net.V = calloc(nb_layers, sizeof(double***));
  net.I = calloc(nb_layers, sizeof(int***));

  if (!net.sizes || !net.a || !net.z || !net.delta || !net.b || !net.W || !net.M || !net.V || !net.I) {
    mlp_free(&net);
    return (MLP){0};
  }

  // 3. Allocate matrix row pointers in bulk (1 allocation each)
  net.W[0] = calloc(row_count, sizeof(double*));
  net.M[0] = calloc(row_count, sizeof(double*));
  net.V[0] = calloc(row_count, sizeof(double*));
  net.I[0] = calloc(row_count, sizeof(int*));

  if (!net.W[0] || !net.M[0] || !net.V[0] || !net.I[0]) {
    mlp_free(&net);
    return (MLP){0};
  }

  // 4. Allocate continuous data blocks in bulk (1 allocation per type)
  net.a[0] = calloc(act_count, sizeof(double));
  net.z[0] = calloc(act_count, sizeof(double));
  net.delta[0] = calloc(act_count, sizeof(double));
  net.b[0] = calloc(bias_count, sizeof(double));
  net.W[0][0] = calloc(weight_count, sizeof(double));
  net.M[0][0] = calloc(weight_count, sizeof(double));
  net.V[0][0] = calloc(weight_count, sizeof(double));
  net.I[0][0] = calloc(weight_count, sizeof(int));

  // Check allocation success
  if (!net.a[0] || !net.z[0] || !net.delta[0] || !net.b[0] || !net.W[0][0] || !net.M[0][0] ||
      !net.V[0][0] || !net.I[0][0]) {
    // Cleanup everything on failure
    mlp_free(&net);
    return (MLP){0};
  }

  // Copy layer sizes
  for (size_t l = 0; l < nb_layers; ++l) {
    net.sizes[l] = sizes[l];
  }

  // 5. Wire up pointer networks without allocating memory in the loops
  size_t act_off = sizes[0];
  size_t bias_off = 0;
  size_t weight_off = 0;
  size_t row_off = 0;



  for (size_t l = 1; l < nb_layers; ++l) {
    size_t count = sizes[l];

    // Assign 1D vector slices from the bulk buffers
    net.a[l] = net.a[0] + act_off;
    net.z[l] = net.z[0] + act_off;
    net.delta[l] = net.delta[0] + act_off;
    act_off += count;

    net.b[l] = net.b[0] + bias_off;
    bias_off += count;

    // Assign 2D matrix row pointers
    net.W[l] = net.W[0] + row_off;
    net.M[l] = net.M[0] + row_off;
    net.V[l] = net.V[0] + row_off;
    net.I[l] = net.I[0] + row_off;

    for (size_t j = 0; j < count; ++j) {
      net.W[l][j] = net.W[0][0] + weight_off + j * NB_ENTRY;
      net.M[l][j] = net.M[0][0] + weight_off + j * NB_ENTRY;
      net.V[l][j] = net.V[0][0] + weight_off + j * NB_ENTRY;
      net.I[l][j] = net.I[0][0] + weight_off + j * NB_ENTRY;
    }

    weight_off += count * NB_ENTRY;
    row_off += count;
  }

  mlp_fill_random(&net);
  return net;
}

void mlp_randomize(MLP* net) {
  if (!net) return;
  mlp_fill_random(net);
}

void mlp_free(MLP* net) {
  if (!net) return;

  if (net->sizes) free(net->sizes);

  if (net->z && net->z[0]) free(net->z[0]);
  if (net->z) free(net->z);

  if (net->a && net->a[0]) free(net->a[0]);
  if (net->a) free(net->a);

  if (net->delta && net->delta[0]) free(net->delta[0]);
  if (net->delta) free(net->delta);

  if (net->b && net->b[0]) free(net->b[0]);
  if (net->b) free(net->b);

  if (net->W && net->W[0] && net->W[0][0]) free(net->W[0][0]);
  if (net->W && net->W[0]) free(net->W[0]);
  if (net->W) free(net->W);

  if (net->M && net->M[0] && net->M[0][0]) free(net->M[0][0]);
  if (net->M && net->M[0]) free(net->M[0]);
  if (net->M) free(net->M);

  if (net->V && net->V[0] && net->V[0][0]) free(net->V[0][0]);
  if (net->V && net->V[0]) free(net->V[0]);
  if (net->V) free(net->V);

  if (net->I && net->I[0] && net->I[0][0]) free(net->I[0][0]);
  if (net->I && net->I[0]) free(net->I[0]);
  if (net->I) free(net->I);

  *net = (MLP){0};
}

void mlp_forward(MLP* net, const double* input) {
  for (unsigned i = 0; i < net->sizes[0]; ++i) {
    net->a[0][i] = input[i];
  }

  for (unsigned l = 1; l < net->nb_layers; ++l) {
    size_t nprev = net->sizes[l - 1];

#pragma omp parallel for schedule(static)
    for (unsigned j = 0; j < net->sizes[l]; ++j) {
      double z = net->b[l][j];

      for (unsigned k = 0; k < NB_ENTRY; ++k) {
        int i = net->I[l][j][k];
        if ((unsigned)i >= (unsigned)nprev) continue;

        double w = weight_from_logit(net, net->W[l][j][k]);
        z += w * net->a[l - 1][i];
      }

      net->z[l][j] = z;
      net->a[l][j] = output_from_sum(net, z);
    }
  }
}

double mlp_backward(MLP* net, const double* target, double lr) {
  int last = net->nb_layers - 1;

  double loss = 0.0;

  // output delta
  for (unsigned j = 0; j < net->sizes[last]; ++j) {
    double y = net->a[last][j];

    double e = y - target[j];

    loss += 0.5 * e * e;

    net->delta[last][j] = e * dtanh_from_output(net, y);
  }

  // hidden deltas
  for (int l = last - 1; l >= 1; --l) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < net->sizes[l]; ++i) {
      double sum = 0.0;

      for (unsigned j = 0; j < net->sizes[l + 1]; ++j) {
        for (int k = 0; k < NB_ENTRY; ++k) {
          if (net->I[l + 1][j][k] == i) {
            double w = weight_from_logit(net, net->W[l + 1][j][k]);

            sum += w * net->delta[l + 1][j];
          }
        }
      }

      net->delta[l][i] = sum * dtanh_from_output(net, net->a[l][i]);
    }
  }

  // update ALL layers
  for (int l = 1; l <= last; ++l) {
#pragma omp parallel for schedule(static)
    for (int j = 0; j < net->sizes[l]; ++j) {
      for (int k = 0; k < NB_ENTRY; ++k) {
        int i = net->I[l][j][k];

        if ((unsigned)i >= (unsigned)net->sizes[l - 1]) continue;

        double grad_w = net->delta[l][j] * net->a[l - 1][i];

        double grad_l = grad_w * dweight_from_logit(net, net->W[l][j][k]);

        net->W[l][j][k] -= lr * grad_l;
      }

      net->b[l][j] -= lr * net->delta[l][j];
    }
  }

  return loss / (double)net->sizes[last];
}

int mlp_save(const MLP* net, const char* path) {
  if (!net || !path) return 0;

  FILE* f = fopen(path, "wb");
  if (!f) return 0;

  uint32_t magic = 0x4D4C5031;  // "MLP1"
  uint32_t version = 1;

  fwrite(&magic, sizeof(magic), 1, f);

  fwrite(&version, sizeof(version), 1, f);

  fwrite(&net->nb_layers, sizeof(size_t), 1, f);

  fwrite(net->sizes, sizeof(size_t), net->nb_layers, f);

  fwrite(&net->weights_temp, sizeof(double), 1, f);

  fwrite(&net->out_temp, sizeof(double), 1, f);

  size_t act_count = 0;     // Total activations across all layers
  size_t bias_count = 0;    // Total biases (layers 1..nb_layers-1)
  size_t weight_count = 0;  // Total weights (layers 1..nb_layers-1)

  for (size_t l = 0; l < net->nb_layers; ++l) {
    act_count += net->sizes[l];
    if (l > 0) {
      bias_count += net->sizes[l];
      weight_count += net->sizes[l] * (size_t)NB_ENTRY;
    }
  }

  fwrite(net->a[0], sizeof(double), act_count, f);
  fwrite(net->z[0], sizeof(double), act_count, f);
  fwrite(net->delta[0], sizeof(double), act_count, f);
  fwrite(net->b[0], sizeof(double), bias_count, f);

  fwrite(net->W[0][0], sizeof(double), weight_count, f);
  fwrite(net->M[0][0], sizeof(double), weight_count, f);
  fwrite(net->V[0][0], sizeof(double), weight_count, f);
  fwrite(net->I[0][0], sizeof(int), weight_count, f);

  fclose(f);
  return 1;
}

MLP mlp_load(const char* path) {
  if (!path) return (MLP){0};

  FILE* f = fopen(path, "rb");
  if (!f) return (MLP){0};

  uint32_t magic = 0;
  uint32_t version = 0;

  // 1. Validate header
  if (fread(&magic, sizeof(magic), 1, f) != 1 || fread(&version, sizeof(version), 1, f) != 1) {
    fclose(f);

    printf("error unvalid file.\n");
    return (MLP){0};
  }

  if (magic != 0x4D4C5031 || version != 1) {
    fclose(f);
    printf("error unvalid vertion.\n");
    return (MLP){0};
  }

  // 2. Read layer count and sizes array
  size_t nb_layers = 0;
  if (fread(&nb_layers, sizeof(size_t), 1, f) != 1 || nb_layers == 0) {
    fclose(f);
    printf("error while reading layers data.\n");
    return (MLP){0};
  }

  size_t* sizes = malloc(nb_layers * sizeof(size_t));
  if (!sizes) {
    fclose(f);
    printf("error while allocating layers data.\n");
    return (MLP){0};
  }

  if (fread(sizes, sizeof(size_t), nb_layers, f) != nb_layers) {
    free(sizes);
    fclose(f);
    printf("error while reading layers data.\n");
    return (MLP){0};
  }

  // 3. Read hyperparameters / temporary settings
  double weights_temp = 0.0;
  double out_temp = 0.0;
  if (fread(&weights_temp, sizeof(double), 1, f) != 1 ||
      fread(&out_temp, sizeof(double), 1, f) != 1) {
    free(sizes);
    fclose(f);
    printf("error while reading temperatures data.\n");
    return (MLP){0};
  }

  // 4. Allocate the network structure using your creation function
  MLP net = mlp_create(nb_layers, sizes, weights_temp, out_temp);
  free(sizes);  // Safe to free temporary sizes buffer now

  if (!net.sizes) {
    fclose(f);
    printf("error allocating data.\n");
    return (MLP){0};
  }

  // 5. Compute total counts matching mlp_save
  size_t act_count = 0;     // Total activations across all layers
  size_t bias_count = 0;    // Total biases (layers 1..nb_layers-1)
  size_t weight_count = 0;  // Total weights (layers 1..nb_layers-1)

  for (size_t l = 0; l < net.nb_layers; ++l) {
    act_count += net.sizes[l];
    if (l > 0) {
      bias_count += net.sizes[l];
      weight_count += net.sizes[l] * (size_t)NB_ENTRY;
    }
  }

  // 6. Read bulk buffers with strict return-value validation
  int read_ok = 1;
  read_ok = read_ok && (fread(net.a[0], sizeof(double), act_count, f) == act_count);
  read_ok = read_ok && (fread(net.z[0], sizeof(double), act_count, f) == act_count);
  read_ok = read_ok && (fread(net.delta[0], sizeof(double), act_count, f) == act_count);
  read_ok = read_ok && (fread(net.b[0], sizeof(double), bias_count, f) == bias_count);

  read_ok = read_ok && (fread(net.W[0][0], sizeof(double), weight_count, f) == weight_count);
  read_ok = read_ok && (fread(net.M[0][0], sizeof(double), weight_count, f) == weight_count);
  read_ok = read_ok && (fread(net.V[0][0], sizeof(double), weight_count, f) == weight_count);
  read_ok = read_ok && (fread(net.I[0][0], sizeof(int), weight_count, f) == weight_count);

  fclose(f);

  if (!read_ok) {
    // If any bulk read failed, you should free the network memory
    // to prevent leaks (e.g., calling mlp_free(&net) if you have a free function).
    printf("error while reading data.\n");
    return (MLP){0};
  }

  return net;
}

struct neurone_truth_generator_data_t {
  double b;
  double* w;
  double temp;
};

static int neurone_truth_generator(void* p, int input) {
  struct neurone_truth_generator_data_t* d = (struct neurone_truth_generator_data_t*)p;

  double z = d->b;
  for (int k = 0; k < NB_ENTRY; ++k) {
    double w = tanh(d->w[k] / d->temp);
    double a = (input >> k & 1) ? 1.0 : -1.0;
    z += w * a;
  }
  return z > 0;
}
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>

#include "aiger.h"

void mlp_aiger_export(const MLP* net, enum aiger_mode mode, FILE* file) {
  aiger* res = aiger_init();

  size_t neuron_count = 0;
  for (unsigned i = 0; i < net->nb_layers; i++) {
    neuron_count += net->sizes[i];
  }

  unsigned** node = malloc(net->nb_layers * sizeof(unsigned*));
  node[0] = malloc(neuron_count * sizeof(unsigned));
  for (unsigned i = 1; i < net->nb_layers; i++) {
    node[i] = node[i - 1] + net->sizes[i - 1];
  }

  unsigned var_count = 1;

  // Initialisation des entrées du réseau (séquentiel)
  for (size_t i = 0; i < net->sizes[0]; i++) {
    node[0][i] = aiger_var2lit(var_count++);
    aiger_add_input(res, node[0][i], NULL);
  }

  // Traitement couche par couche (la dépendance entre couches impose un ordre séquentiel)
  for (unsigned l = 1; l < net->nb_layers; l++) {
// Parallélisation au niveau des neurones de la couche l
#pragma omp parallel
    {
      struct neurone_truth_generator_data_t data;
      data.temp = net->weights_temp;

#pragma omp for schedule(dynamic)
      for (unsigned i = 0; i < net->sizes[l]; i++) {
        data.b = net->b[l][i];
        data.w = net->W[l][i];

        qm16_vec_t qm = {0};
        // QM16 est thread-safe car data et qm sont locaux au thread
        qm16_generate(&qm, neurone_truth_generator, &data);

        unsigned current_node = aiger_false;

        for (size_t term_id = 0; term_id < qm.size; term_id++) {
          uint32_t term = qm.data[term_id];
          uint16_t mask = (uint16_t)term;
          uint16_t value = (uint16_t)(term >> 16);

          unsigned imp = aiger_true;

          for (int input_id = 0; input_id < NB_ENTRY; ++input_id) {
            if (mask & (1u << input_id)) continue;

            // Lecture sécurisée : la couche (l-1) est déjà entièrement calculée
            unsigned lit = node[l - 1][net->I[l][i][input_id]];
            if (!(value & (1u << input_id))) {
              lit = aiger_not(lit);
            }

            if (imp == aiger_true) {
              imp = lit;
            } else {
              // Section critique pour la création de variable et l'ajout de porte AIGER
              unsigned new_imp;
#pragma omp critical(aiger_mutation)
              {
                new_imp = aiger_var2lit(var_count++);
                aiger_add_and(res, new_imp, imp, lit);
              }
              imp = new_imp;
            }
          }

          if (current_node == aiger_false) {
            current_node = imp;
          } else {
            // Section critique pour l'assemblage des termes (De Morgan)
            unsigned new_gate;
#pragma omp critical(aiger_mutation)
            {
              new_gate = aiger_var2lit(var_count++);
              aiger_add_and(res, new_gate, aiger_not(current_node), aiger_not(imp));
            }
            current_node = aiger_not(new_gate);
          }
        }

        // Affectation finale du résultat du neurone (i)
        node[l][i] = current_node;
      }
    }
  }

  // Déclaration des sorties (séquentiel)
  for (size_t i = 0; i < net->sizes[net->nb_layers - 1]; i++) {
    aiger_add_output(res, node[net->nb_layers - 1][i], NULL);
  }

  aiger_check(res);
  aiger_write_to_file(res, mode, file);

  free(node[0]);
  free(node);
  aiger_reset(res);
}