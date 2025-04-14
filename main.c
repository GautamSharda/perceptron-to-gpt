#include <stdio.h>
#include <math.h>   // Include for ceil()
#include <stdlib.h> // Include for malloc() and free()

typedef struct{
    double value;
    double gradient;
} Value;

typedef struct{
    Value weight;
    Value bias;
} Neuron;

typedef struct {
    double x;
    double y;
} DataPoint;

const DataPoint DATA[] = {
    {0.0, 0.0},
    {1.0, 1.0},
    {2.0, 2.0},
    {3.0, 3.0},
    {4.0, 4.0},
    {5.0, 5.0},
    {6.0, 6.0},
    {7.0, 7.0},
    {8.0, 8.0},
    {9.0, 9.0}
};

typedef struct {
    double *epoch_losses;
    double final_weight;
    double final_bias;
    double final_avg_loss;
    int best_epoch;
    int epochs;
} Results;

typedef struct {
    DataPoint *datapoints;
} Batch;

const int DATA_SIZE = 10;


double forward(Neuron *neuron, double x){
    return neuron->weight.value*x + neuron->bias.value; // BUT I'd prefer to implicitly build the compute graph here
}

void gradient_descent(int epochs, DataPoint *data_array, int data_size, int batch_size, int accum){
    Results result;
    result.epochs = epochs;
    result.epoch_losses = NULL; // This should be allocated if needed
    result.final_weight = 0.0;
    result.final_bias = 0.0;
    result.final_avg_loss = 0.0;
    result.best_epoch = 0;

    double num_batches = ceil((double)data_size / batch_size);
    Batch *batches = (Batch *)malloc(num_batches * sizeof(Batch));
    for (int d = 0; d < data_size; d++){
        DataPoint *datapoints = (DataPoint *)malloc(batch_size*sizeof(DataPoint));
        datapoints[d] = data_array[d];
        if (d % batch_size == 0){
            Batch batch;
            batch.datapoints = datapoints;
            batches[d / batch_size] = batch;
        }
    }
    // Print batches information
    printf("Number of batches: %.0f\n", num_batches);
    for (int i = 0; i < (int)num_batches; i++) {
        printf("Batch %d:\n", i);
        for (int j = 0; j < batch_size; j++) {
            // Check if we're still within data bounds
            int data_index = i * batch_size + j;
            if (data_index < data_size) {
                printf("  DataPoint %d: x=%.2f, y=%.2f\n", 
                       data_index, 
                       batches[i].datapoints[j].x, 
                       batches[i].datapoints[j].y);
            }
        }
    }
    
    for (int i = 0; i < (int)num_batches; i++) {
        free(batches[i].datapoints);
    }
    free(batches);
    // result;
}

int main(){
    Value weight;
    weight.value = 20.0;
    weight.gradient = 1.0;

    Value bias;
    bias.value = 10.0;
    bias.gradient = 0.0;

    Neuron neuron;
    neuron.weight = weight;
    neuron.bias = bias;

    gradient_descent(1, DATA, 10, 3, 1);

    // for (int i = 0; i < DATA_SIZE; i++){
    //     double loss = forward(&neuron, DATA[i].x) - DATA[i].y;
    //     printf("%f\n", loss);
    // }
}