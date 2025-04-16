#include <stdio.h>
#include <math.h>
#include <stdlib.h>

typedef struct Value {
    double value;
    double gradient;
    int op; // 0 = add, 1 = mul, 2 = sub, 3 = div, 4 = abs, -1 = default / root
    int right;
    struct Value *children[2];
} Value;

// define and use constructor

Value add_values(Value *val_1, Value *val_2){
    val_1->op = 0;
    val_2->op = 0;
    val_2->right = 1;
    Value result;
    result.value = val_1->value + val_2->value;
    result.gradient = 0;
    result.op = -1;
    result.right = 0;
    result.children[0] = val_1;
    result.children[1] = val_2;
    return result;
}

Value mul_values(Value *val_1, Value *val_2){
    val_1->op = 1;
    val_2->op = 1;
    val_2->right = 1;
    Value result;
    result.value = val_1->value * val_2->value;
    result.gradient = 0;
    result.op = -1;
    result.right = 0;
    result.children[0] = val_1;
    result.children[1] = val_2;
    return result;
}

Value sub_values(Value *val_1, Value *val_2){
    val_1->op = 2;
    val_2->op = 2;
    val_2->right = 1;
    Value result;
    result.value = val_1->value - val_2->value;
    result.gradient = 0;
    result.op = -1;
    result.right = 0;
    result.children[0] = val_1;
    result.children[1] = val_2;
    return result;
}

Value div_values(Value *val_1, Value *val_2){
    val_1->op = 3;
    val_2->op = 3;
    val_2->right = 1;
    Value result;
    result.value = val_1->value / val_2->value;
    result.gradient = 0;
    result.op = -1;
    result.right = 0;
    result.children[0] = val_1;
    result.children[1] = val_2;
    return result;
}

Value fabs_value(Value *val_1){
    val_1->op = 4;
    Value result;
    result.value = fabs(val_1->value);
    result.gradient = 0;
    result.op = -1;
    result.right = 0;
    result.children[0] = val_1;
    result.children[1] = NULL;
    return result;
}

void backward(Value *v, Value *other, double chain_rule_grad){
    switch (v->op) {
        // 0 = add, 1 = mul, 2 = sub, 3 = div, 4 = abs
        case 0:
            v->gradient += 1.0*chain_rule_grad;
            break;
        case 1:
            v->gradient += other->value*chain_rule_grad;
            break;
        case 2:
            if (v->right) {
                v->gradient += -1.0*chain_rule_grad;
            } else {
                v->gradient += 1.0*chain_rule_grad;
            }
            break;
        case 3:
            if (v->right) {
                v->gradient += ((-other->value) / (v->value * v->value)) * chain_rule_grad;
            } else {
                v->gradient += (1.0 / other->value) * chain_rule_grad;
            }
            break;
        case 4:
            v->gradient += ((v->value < 0) ? -1.0 : 1.0) * chain_rule_grad;
            break;
        default: // root
            v->gradient += 1.0;
            break;
    }

    if (v->children[0] != NULL) {
        backward(v->children[0], v->children[1], v->gradient);
    }
    if (v->children[1] != NULL) {
        backward(v->children[1], v->children[0], v->gradient);
    }
}

typedef struct{
    Value weight;
    Value bias;
    double lr;
} Neuron;

Value forward(Neuron *neuron, Value *x){
    Value mul_result = mul_values(&neuron->weight, x);
    return add_values(&mul_result, &neuron->bias);
}

void descend(Neuron *n){
    n->weight.value = n->weight.value - n->lr*n->weight.gradient;
    n->bias.value = n->bias.value - n->lr*n->bias.gradient;
    n->weight.gradient = 0.0;
    n->bias.gradient = 0.0;
}

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
    double best_loss;
    int best_epoch;
    int epochs;
} Results;

typedef struct {
    DataPoint *datapoints;
} Batch;

const int DATA_SIZE = 10;

typedef struct {
    double *d;
    int size;
} DoubleArray;

Results gradient_descent(Neuron *n, int epochs, DataPoint *data_array, int data_size, int batch_size, int accum){
    Results result;
    result.epochs = epochs;
    result.epoch_losses = NULL; // This should be allocated if needed
    result.final_weight = 0.0;
    result.final_bias = 0.0;
    result.final_avg_loss = 0.0;
    result.best_loss = 0.0;
    result.best_epoch = 0;

    int num_batches = (int)ceil((double) data_size / batch_size);
    Batch *batches = (Batch *)malloc(num_batches * sizeof(Batch));
    for (int b = 0; b < num_batches; b++){
        DataPoint *datapoints = (DataPoint *)malloc(batch_size*sizeof(DataPoint));
        int batch_count = 0;
        for(int d = b*batch_size; d < data_size; d++){
            if (batch_count == batch_size){
                break;
            }
            datapoints[batch_count] = data_array[d];
            batch_count++;
        }
        Batch batch;
        batch.datapoints = datapoints;
        batches[b] = batch;
    }

    double epoch_loss = 0.0;
    double *epoch_losses = malloc(epochs*sizeof(double));
    for (int e = 0; e < epochs; e++){
        int accum_count = 0;
        for (int b = 0; b < num_batches; b++){
            printf("Batch %d:\n", b);
            Value batch_loss;
            batch_loss.value = 0.0;
            batch_loss.gradient = 0.0;
            batch_loss.op = -1;
            batch_loss.right = 0;
            for (int d = 0; d < batch_size; d++){
                int data_index = b * batch_size + d;
                if (data_index >= data_size) {
                    break;
                }
                printf("  DataPoint: x=%.2f, y=%.2f\n", 
                        batches[b].datapoints[d].x, 
                        batches[b].datapoints[d].y);
                Value x_val;
                x_val.value = batches[b].datapoints[d].x;
                x_val.gradient = 0.0;
                x_val.op = -1;
                x_val.right = 0;
                Value y_val;
                y_val.value = batches[b].datapoints[d].y;
                y_val.gradient = 0.0;
                y_val.op = -1;
                y_val.right = 0;

                Value prediction = forward(n, &x_val);
                Value diff = sub_values(&prediction, &y_val);
                Value loss_term = fabs_value(&diff);
                batch_loss = add_values(&batch_loss, &loss_term);
            }
            epoch_loss += batch_loss.value;

            Value batch_size_val;
            batch_size_val.value = batch_size;
            batch_size_val.gradient = 0.0;
            batch_size_val.op = -1;
            batch_size_val.right = 0;
            Value avg_loss = div_values(&batch_loss, &batch_size_val);
            backward(&avg_loss, NULL, 1.0);
            accum_count += 1;
            if (accum_count % accum == 0){
                descend(n);
            }
        }
        epoch_losses[e] = epoch_loss / num_batches;
    }
    result.epoch_losses = epoch_losses;
    result.final_weight = n->weight.value;
    result.final_bias = n->bias.value;
    result.final_avg_loss = epoch_losses[epochs-1];

    // linear scan to find min
    double curr_min = epoch_losses[0];
    int best_epoch = 1;
    for (int e = 1; e < epochs; e++){
        if (curr_min > epoch_losses[e]){
            curr_min = epoch_losses[e];
            best_epoch = e + 1;
        }
    }
    result.best_loss = curr_min;
    result.best_epoch = best_epoch;

    for (int b = 0; b < num_batches; b++){
        free(batches[b].datapoints);
    }
    free(batches);

    return result;
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
    neuron.lr = 0.025;

    Results result = gradient_descent(&neuron, 1, (DataPoint *) DATA, 10, 3, 1);
    // Print the results
    printf("Training Results:\n");
    printf("----------------\n");
    printf("Final Weight: %.4f\n", result.final_weight);
    printf("Final Bias: %.4f\n", result.final_bias);
    printf("Best Loss: %.4f (at epoch %d)\n", result.best_loss, result.best_epoch);
    free(result.epoch_losses);    
}