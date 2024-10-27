#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <thread>

typedef struct{
    int N;
    int term;
    double* x;
    double* y;
}sin_args;

void sinx(int N, int term, double* x, double* y){

    double numer;
    double denom;
    double value;
    int sign;

    for (int i=0; i<N; i++){
        numer = x[i]* x[i] * x[i];
        denom = 6;
        sign = -1;

        value = x[i];

        for (int j = 1; j < term; j++){
            value += sign * numer/denom;
            numer *= x[i] * x[i];
            denom *= (2*j + 2) * (2*j + 3); 
            sign  *= sign;
        }

        y[i] = value;
    }
}

void thread_func(sin_args* args){
    sinx(args->N, args->term, args->x, args->y);
}

int main (){
    int N = 10;
    int term = 3;

    double x[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    double* y = new double[N];

    sin_args args;
    args.N = N/2;
    args.term = term;
    args.x = x;
    args.y = y;

    std::thread tsin1;

    tsin1 = std::thread(thread_func, &args);
    sinx(N/2, term, x+N/2 , y+N/2);
    tsin1.join();

    for (int i=0; i<N; i++){
        std::cout<<"Value is: "<<y[i]<<std::endl;
    }
    
}