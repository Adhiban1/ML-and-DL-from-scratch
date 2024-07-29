#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define in_shape 10
#define h1 20
#define h2 10
#define out_shape 50
#define n 10

#define upper 1
#define lower -1

void rand_init(float *x, int r, int c) {
    #pragma omp parallel for
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            x[i * c + j] = (float)rand()/RAND_MAX * (upper - lower) + lower;
}

void print_matrix(float *x, int r, int c) {
    #pragma omp parallel for
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++)
            printf("%f ", x[i * c + j]);
        printf("\n");
    }
    printf("\n");
}

void dot(float *a3, float *a1, float *a2, int r, int c, int c2) {
    #pragma omp parallel for
    for (int i = 0; i < r; i++) {
        for (int k = 0 ; k < c2; k++) {
            float temp = 0;
            for (int j = 0; j < c; j++) {
                temp += a1[i * c + j] * a2[j * c2 + k];
            }
            a3[i * c2 + k] = temp;
        }
    }
}

void forward(float* x, float* w1, float* w2, float* w3, float* y) {
    float *z1 = (float*)malloc(n * h1 * sizeof(float));
    float *z2 = (float*)malloc(n * h2 * sizeof(float));

    if (z1 == NULL || z2 == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        return;
    }

    dot(z1, x, w1, n, in_shape, h1);
    // printf("z1 [Done]\n");
    dot(z2, z1, w2, n, h1, h2);
    // printf("z2 [Done]\n");
    dot(y, z2, w3, n, h2, out_shape);
    // printf("y [Done]\n");

    free(z1);
    free(z2);
}

float loss(float* yt, float* y) {
    float l = 0, temp;
    #pragma omp parallel for
    for (int i = 0; i < n * out_shape; i++) {
        temp = yt[i] - y[i];
        l += temp * temp;
    }
    l /= n * out_shape;
    return l;
}

void copy(float* a, float* b, int l) {
    #pragma omp parallel for
    for (int i = 0; i < l; i++) {
        if (a[i] != b[i])
            a[i] = b[i];
    }
}

int main() {
    srand(time(NULL));

    float *x = (float*)malloc(n * in_shape * sizeof(float));
    float *w1 = (float*)malloc(in_shape * h1 * sizeof(float));
    float *w2 = (float*)malloc(h1 * h2 * sizeof(float));
    float *w3 = (float*)malloc(h2 * out_shape * sizeof(float));
    float *dw1 = (float*)malloc(in_shape * h1 * sizeof(float));
    float *dw2 = (float*)malloc(h1 * h2 * sizeof(float));
    float *dw3 = (float*)malloc(h2 * out_shape * sizeof(float));
    float *tw1 = (float*)malloc(in_shape * h1 * sizeof(float));
    float *tw2 = (float*)malloc(h1 * h2 * sizeof(float));
    float *tw3 = (float*)malloc(h2 * out_shape * sizeof(float));
    float *t2w1 = (float*)malloc(in_shape * h1 * sizeof(float));
    float *t2w2 = (float*)malloc(h1 * h2 * sizeof(float));
    float *t2w3 = (float*)malloc(h2 * out_shape * sizeof(float));
    float *y1 = (float*)malloc(n * out_shape * sizeof(float));
    float *y2 = (float*)malloc(n * out_shape * sizeof(float));
    float *yt = (float*)malloc(n * out_shape * sizeof(float));

    float h = 0.001, lr = 0.001, l1, l2, l;

    // if (x == NULL || w1 == NULL || w2 == NULL || w3 == NULL || y1 == NULL) {
    //     fprintf(stderr, "Memory allocation failed!\n");
    //     return 1;
    // }

    rand_init(x, n, in_shape);
    rand_init(w1, in_shape, h1);
    rand_init(w2, h1, h2);
    rand_init(w3, h2, out_shape);
    rand_init(yt, n, out_shape);

    forward(x, w1, w2, w3, y1);
    l = loss(yt, y1);
    printf("Loss: %f\n", l);

    for (int epoch = 0; epoch < 1000; epoch++) {
        copy(tw1, w1, in_shape * h1);
        copy(t2w1, w1, in_shape * h1);
        copy(tw2, w2, in_shape * h1);
        copy(t2w2, w2, in_shape * h1);
        copy(tw3, w3, in_shape * h1);
        copy(t2w3, w3, in_shape * h1);

        #pragma omp parallel for
        for (int i = 0; i < in_shape * h1; i++) {
            tw1[i] = tw1[i] + h;
            t2w1[i] = t2w1[i] - h;
            forward(x, tw1, w2, w3, y1);
            forward(x, t2w1, w2, w3, y2);
            l1 = loss(yt, y1);
            l2 = loss(yt, y2);
            dw1[i] = (l1 - l2) / (2 * h);
            tw1[i] = w1[i];
            t2w1[i] = w1[i];
        }
        // printf("dw1 [Done]\n");

        #pragma omp parallel for
        for (int i = 0; i < h1 * h2; i++) {
            tw2[i] = tw2[i] + h;
            t2w2[i] = t2w2[i] - h;
            forward(x, w1, tw2, w3, y1);
            forward(x, w1, t2w2, w3, y2);
            l1 = loss(yt, y1);
            l2 = loss(yt, y2);
            dw2[i] = (l1 - l2) / (2 * h);
            tw2[i] = w2[i];
            t2w2[i] = w2[i];
        }
        // printf("dw2 [Done]\n");

        #pragma omp parallel for
        for (int i = 0; i < h2 * out_shape; i++) {
            tw3[i] = tw3[i] + h;
            t2w3[i] = t2w3[i] - h;
            forward(x, w1, w2, tw3, y1);
            forward(x, w1, w2, t2w3, y2);
            l1 = loss(yt, y1);
            l2 = loss(yt, y2);
            dw3[i] = (l1 - l2) / (2 * h);
            tw3[i] = w3[i];
            t2w3[i] = w3[i];
        }
        // printf("dw3 [Done]\n");

        #pragma omp parallel for
        for (int i = 0; i < in_shape * h1; i++)
            w1[i] -= dw1[i] * lr;

        #pragma omp parallel for
        for (int i = 0; i < h1 * h2; i++)
            w2[i] -= dw2[i] * lr;
        
        #pragma omp parallel for
        for (int i = 0; i < h2 * out_shape; i++)
            w3[i] -= dw3[i] * lr;

        // printf("w update [Done]\n");

        forward(x, w1, w2, w3, y1);
        l = loss(yt, y1);
        printf("Epoch: %d, Loss: %f\n", epoch + 1, l);
    }

    // Free allocated memory
    free(x);
    free(w1);
    free(w2);
    free(w3);
    free(dw1);
    free(dw2);
    free(dw3);
    free(tw1);
    free(tw2);
    free(tw3);
    free(t2w1);
    free(t2w2);
    free(t2w3);
    free(y1);
    free(y2);

    return 0;
}
