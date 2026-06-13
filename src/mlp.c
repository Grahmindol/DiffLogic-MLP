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

static size_t align_up(size_t x, size_t a) { return (x + a - 1) & ~(a - 1); }

static size_t append(size_t* off, size_t align, size_t bytes) {
  *off = align_up(*off, align);
  size_t pos = *off;
  *off += bytes;
  return pos;
}

MLP mlp_create(size_t nb_layers, const size_t* sizes, double weights_temp, double out_temp) {
  MLP net = {0};

  if (nb_layers < 1) return net;

  net.nb_layers = nb_layers;
  net.weights_temp = (weights_temp < 0.05) ? 0.05 : weights_temp;
  net.out_temp = (out_temp < 0.05) ? 0.05 : out_temp;

  size_t act_count = 0;
  size_t bias_count = 0;
  size_t weight_count = 0;
  size_t row_count = 0;

  for (unsigned l = 0; l < nb_layers; ++l) {
    act_count += sizes[l];
    if (l > 0) {
      bias_count += sizes[l];
      weight_count += sizes[l] * (size_t)NB_ENTRY;
      row_count += sizes[l];
    }
  }

  size_t moff = 0;
  size_t sizes_off = append(&moff, sizeof(size_t), nb_layers * sizeof(size_t));
  size_t a_off = append(&moff, sizeof(double*), nb_layers * sizeof(double*));
  size_t z_off = append(&moff, sizeof(double*), nb_layers * sizeof(double*));
  size_t d_off = append(&moff, sizeof(double*), nb_layers * sizeof(double*));
  size_t b_off = append(&moff, sizeof(double*), nb_layers * sizeof(double*));
  size_t W_off = append(&moff, sizeof(double**), nb_layers * sizeof(double**));
  size_t I_off = append(&moff, sizeof(int**), nb_layers * sizeof(int**));

  size_t Wrow_off = append(&moff, sizeof(double*), row_count * sizeof(double*));
  size_t Irow_off = append(&moff, sizeof(int*), row_count * sizeof(int*));

  net.meta_block = calloc(1, moff);
  if (!net.meta_block) return (MLP){0};

  net.data_block = (double*)calloc(1, (3 * act_count + bias_count + weight_count) * sizeof(double));
  if (!net.data_block) {
    free(net.meta_block);
    return (MLP){0};
  }

  net.idx_block = (int*)calloc(1, weight_count * sizeof(int));
  if (!net.idx_block) {
    free(net.meta_block);
    free(net.data_block);
    return (MLP){0};
  }

  unsigned char* m = (unsigned char*)net.meta_block;

  net.sizes = (size_t*)(m + sizes_off);
  net.a = (double**)(m + a_off);
  net.z = (double**)(m + z_off);
  net.delta = (double**)(m + d_off);
  net.b = (double**)(m + b_off);
  net.W = (double***)(m + W_off);
  net.I = (int***)(m + I_off);

  double** Wrows = (double**)(m + Wrow_off);
  int** Irows = (int**)(m + Irow_off);

  for (unsigned l = 0; l < nb_layers; ++l) {
    net.sizes[l] = sizes[l];
  }

  double* A = net.data_block;
  double* Z = A + act_count;
  double* D = Z + act_count;
  double* B = D + act_count;
  double* Wv = B + bias_count;

  int* Iv = net.idx_block;

  size_t a_ofs = 0;
  size_t b_ofs = 0;
  size_t w_ofs = 0;
  size_t r_ofs = 0;

  for (unsigned l = 0; l < nb_layers; ++l) {
    net.a[l] = A + a_ofs;
    net.z[l] = Z + a_ofs;
    net.delta[l] = D + a_ofs;
    a_ofs += sizes[l];

    if (l == 0) {
      net.b[l] = NULL;
      net.W[l] = NULL;
      net.I[l] = NULL;
      continue;
    }

    net.b[l] = B + b_ofs;
    b_ofs += sizes[l];

    net.W[l] = Wrows + r_ofs;
    net.I[l] = Irows + r_ofs;

    for (size_t j = 0; j < sizes[l]; ++j) {
      net.W[l][j] = Wv + w_ofs + j * NB_ENTRY;
      net.I[l][j] = Iv + w_ofs + j * NB_ENTRY;
    }

    w_ofs += sizes[l] * NB_ENTRY;
    r_ofs += sizes[l];
  }

  mlp_fill_random(&net);
  return net;
}

void mlp_randomize(MLP* net) {
  if (!net || !net->meta_block) return;
  mlp_fill_random(net);
}

void mlp_free(MLP* net) {
  if (!net) return;
  free(net->meta_block);
  free(net->data_block);
  free(net->idx_block);
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

  size_t act_count = 0;
  size_t bias_count = 0;
  size_t weight_count = 0;

  for (unsigned l = 0; l < net->nb_layers; ++l) {
    act_count += net->sizes[l];

    if (l > 0) {
      bias_count += net->sizes[l];

      weight_count += net->sizes[l] * NB_ENTRY;
    }
  }

  size_t data_count = 3 * act_count + bias_count + weight_count;

  fwrite(&data_count, sizeof(size_t), 1, f);

  fwrite(net->data_block, sizeof(double), data_count, f);

  fwrite(&weight_count, sizeof(size_t), 1, f);

  fwrite(net->idx_block, sizeof(int), weight_count, f);

  fclose(f);
  return 1;
}

MLP mlp_load(const char* path) {
  FILE* f = fopen(path, "rb");

  if (!f) return (MLP){0};

  uint32_t magic;
  uint32_t version;

  fread(&magic, sizeof(magic), 1, f);

  fread(&version, sizeof(version), 1, f);

  if (magic != 0x4D4C5031 || version != 1) {
    fclose(f);
    return (MLP){0};
  }

  size_t nb_layers;

  fread(&nb_layers, sizeof(size_t), 1, f);

  size_t* sizes = malloc(nb_layers * sizeof(size_t));

  fread(sizes, sizeof(size_t), nb_layers, f);

  double weights_temp;
  double out_temp;

  fread(&weights_temp, sizeof(double), 1, f);

  fread(&out_temp, sizeof(double), 1, f);

  MLP net = mlp_create(nb_layers, sizes, weights_temp, out_temp);

  free(sizes);

  if (!net.meta_block) {
    fclose(f);
    return (MLP){0};
  }

  size_t data_count;

  fread(&data_count, sizeof(size_t), 1, f);

  fread(net.data_block, sizeof(double), data_count, f);

  size_t weight_count;

  fread(&weight_count, sizeof(size_t), 1, f);

  fread(net.idx_block, sizeof(int), weight_count, f);

  fclose(f);

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

void mlp_aiger_export(const MLP* net, enum aiger_mode mode, FILE* file) {
  aiger* res = aiger_init();

  size_t neuron_count = 0;
  for (unsigned i = 0; i < net->nb_layers; i++) neuron_count += net->sizes[i];

  unsigned** node = malloc(net->nb_layers * sizeof(unsigned*));
  node[0] = malloc(neuron_count * sizeof(unsigned));
  for (unsigned i = 1; i < net->nb_layers; i++) node[i] = node[i - 1] + net->sizes[i - 1];

  unsigned var_count = 1;

  for (size_t i = 0; i < net->sizes[0]; i++) {
    node[0][i] = aiger_var2lit(var_count++);
    aiger_add_input(res, node[0][i], NULL);
  }

  struct neurone_truth_generator_data_t data;
  data.temp = net->weights_temp;
  for (unsigned l = 1; l < net->nb_layers; l++) {
    for (unsigned i = 0; i < net->sizes[l]; i++) {
      data.b = net->b[l][i];
      data.w = net->W[l][i];
      qm16_vec_t qm = {0};
      qm16_generate(&qm, neurone_truth_generator, &data);

      node[l][i] = aiger_false;
      for (size_t term_id = 0; term_id < qm.size; term_id++) {
        uint32_t term = qm.data[term_id];
        uint16_t mask = (uint16_t)term;
        uint16_t value = (uint16_t)(term >> 16);

        unsigned imp = aiger_true;
        for (int input_id = 0; input_id < NB_ENTRY; ++input_id) {
          if (mask & (1u << input_id)) continue;

          unsigned lit = node[l - 1][net->I[l][i][input_id]];
          if (!(value & (1u << input_id))) {
            lit = aiger_not(lit);
          }

          if (imp == aiger_true) {
            imp = lit;
          } else {
            unsigned new_imp = aiger_var2lit(var_count++);
            aiger_add_and(res, new_imp, imp, lit);
            imp = new_imp;
          }
        }

        if (node[l][i] == aiger_false) {
          node[l][i] = imp;
        } else {
          unsigned new_gate = aiger_var2lit(var_count++);
          aiger_add_and(res, new_gate, aiger_not(node[l][i]), aiger_not(imp));
          node[l][i] = aiger_not(new_gate);
        }
      }
    }
  }

  for (size_t i = 0; i < net->sizes[net->nb_layers - 1]; i++) {
    aiger_add_output(res, node[net->nb_layers - 1][i], NULL);
  }

  aiger_check(res);
  aiger_write_to_file(res, mode, file);

  free(node[0]);
  free(node);
  aiger_reset(res);
}