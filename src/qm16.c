#include "qm16.h"

#include <stdio.h>
#include <string.h>

/* =========================
   bit encoding
   =========================
   [31..16] value
   [15.. 0] mask
*/

static inline uint16_t mask(uint32_t x) { return (uint16_t)x; }
static inline uint16_t value(uint32_t x) { return (uint16_t)(x >> 16); }

static inline int popc(uint32_t x) { return __builtin_popcount(value(x) & ~mask(x)); }

/* reduce */
static int reduce(uint32_t a, uint32_t b, uint32_t* out) {
  if (mask(a) != mask(b)) return 0;

  uint16_t va = value(a);
  uint16_t vb = value(b);

  uint16_t diff = va ^ vb;

  if (diff == 0 || (diff & (diff - 1))) return 0;

  *out = ((uint32_t)(va & ~diff) << 16) | (mask(a) | diff);

  return 1;
}

/* =========================
   vector utils
   ========================= */

static int push(qm16_vec_t* v, uint32_t x) {
  if (v->size >= v->capacity) {
    size_t nc = v->capacity ? v->capacity * 2 : 16;
    uint32_t* p = realloc(v->data, nc * sizeof(uint32_t));
    if (!p) return 0;
    v->data = p;
    v->capacity = nc;
  }
  v->data[v->size++] = x;
  return 1;
}

void qm16_free(qm16_vec_t* v) {
  free(v->data);
  v->data = NULL;
  v->size = 0;
  v->capacity = 0;
}

/* =========================
   core derive (simple QM)
   ========================= */

static int derive(qm16_vec_t* in, qm16_vec_t* out, qm16_vec_t* primes) {
  int changed = 0;
  uint8_t* used = calloc(in->size, 1);

  out->size = 0;

  for (size_t i = 0; i < in->size; i++) {
    for (size_t j = i + 1; j < in->size; j++) {
      uint32_t r;
      if (reduce(in->data[i], in->data[j], &r)) {
        used[i] = 1;
        used[j] = 1;
        changed = 1;

        int exists = 0;
        for (size_t k = 0; k < out->size; k++)
          if (out->data[k] == r) exists = 1;

        if (!exists) push(out, r);
      }
    }
  }

  for (size_t i = 0; i < in->size; i++) {
    if (!used[i]) push(primes, in->data[i]);
  }

  free(used);
  return changed;
}

static void derive_all(qm16_vec_t* in, qm16_vec_t* primes) {
  qm16_vec_t cur = {0}, next = {0};

  cur.data = malloc(in->size * sizeof(uint32_t));
  memcpy(cur.data, in->data, in->size * sizeof(uint32_t));
  cur.size = in->size;
  cur.capacity = in->size;

  while (1) {
    if (!derive(&cur, &next, primes)) break;

    qm16_vec_t tmp = cur;
    cur = next;
    next = tmp;

    next.size = 0;
  }

  free(cur.data);
  free(next.data);
}

/* =========================
   API
   ========================= */

int qm16_generate(qm16_vec_t* out, int (*gen)(void*, int), void* user_data) {
  qm16_vec_t minterms = {0};
  qm16_vec_t primes = {0};

  for (uint32_t i = 0; i < QM16_SPACE; i++) {
    if (gen(user_data, i)) {
      uint32_t term = i << 16;
      push(&minterms, term);
    }
  }

  derive_all(&minterms, &primes);

  qm16_free(out);
  *out = primes;

  qm16_free(&minterms);
  return 1;
}