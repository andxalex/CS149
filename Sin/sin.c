#include <stdio.h>
#include <stdlib.h>

void sinx(int N, int terms, double *x, double *y) {
  for (int i = 0; i < N; i++) {
    double value = x[i];

    // Compute fraction
    double numer = x[i] * x[i] * x[i];
    double denom = 6; // 3!
    int sign = -1;

    for (int j = 1; j < terms; j++) {
      value += numer / denom * sign;
      numer *= x[i] * x[i];
      denom *= (2 * j + 2) * (2 * j + 3);
      sign *= -1;
    }

    y[i] = value;
  }
}

int main() {
  int N = 1;      // one value
  int terms = 3; // 150 terms

  double x = 0.1;
  double *y = malloc(sizeof(double) * N);

  sinx(N, terms, &x, y);

  printf("Value is: %f", *y);

  free(y);
}