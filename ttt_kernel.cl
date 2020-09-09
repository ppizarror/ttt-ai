#define pop_amount 150

typedef struct
{
    // All the weights.
    float* w;
    // Hidden to output layer weights.
    float* x;
    // Biases.
    float* b;
    // Hidden layer.
    float* h;
    // Output layer.
    float* o;
    // Number of biases - always two - Tinn only supports a single hidden layer.
    int nb;
    // Number of weights.
    int nw;
    // Number of inputs.
    int nips;
    // Number of hidden neurons.
    int nhid;
    // Number of outputs.
    int nops;
} Tinn;

static float act(const float a)
{
    return 1.0f / (1.0f + exp(-a));
}

static void fprop(const Tinn t, const float* const in)
{
    // Calculate hidden layer neuron values.
    for(int i = 0; i < t.nhid; i++)
    {
        float sum = 0.0f;
        for(int j = 0; j < t.nips; j++)
            sum += in[j] * t.w[i * t.nips + j];
        t.h[i] = act(sum + t.b[0]);
    }
    // Calculate output layer neuron values.
    for(int i = 0; i < t.nops; i++)
    {
        float sum = 0.0f;
        for(int j = 0; j < t.nhid; j++)
            sum += t.h[j] * t.x[i * t.nhid + j];
        t.o[i] = act(sum + t.b[1]);
    }
}

float* xtpredict(const Tinn t, const float* const in)
{
    fprop(t, in);
    return t.o;
}

typedef struct ans{
    int id;
    int fitness;
    Tinn red;
} ans;

bool isMovesLeft(char board[3][3]) 
{ 
    for (int i = 0; i<3; i++) 
        for (int j = 0; j<3; j++) 
            if (board[i][j]=='_') 
                return true;
    return false; 
}

int randomPlay(char board[3][3], int random[20])
{
    bool existe = false;
    int ans = -1;
    for(int i=0;i<20;i++){
        int pos_x = random[i]%3;
        int pos_y = random[i]/3;
        if(board[pos_y][pos_x] == '_'){
            existe = true;
            ans = random[i];
            break;
        }
    }

    if(existe){
        return ans;
    }
    else{
        int it = 0;
        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                if(board[i][j]=='_'){
                    ans = it;
                    return ans;
                }
                it++;
            }
        }
    }
    
}


__kernel void match(__global const Tinn *ann, __global ans *respuestas, __global char all_boards[pop_amount][3][3], __global int random[pop_amount][20]){
    
    //get id
    const int id = get_global_id(0);
    Tinn kernel_ann = ann[id];
    int correct_choices = 0;

    //set parameters
    char board[3][3] = {
        {all_boards[id][0][0],all_boards[id][0][1],all_boards[id][0][2]},
        {all_boards[id][1][0],all_boards[id][1][1],all_boards[id][1][2]},
        {all_boards[id][2][0],all_boards[id][2][1],all_boards[id][2][2]}
    };

    int id_random[20];
    for(int i=0;i<20;i++){
        id_random[i] = random[id][i];
    }
    
    char sign[3] = {'x', 'o', '_'};
    int turns = 0;

    while(isMovesLeft(board))
    {
        //build network's input
        int glob = 0;
        float input[27];

        for(int k=0; k<3;k++){
            for(int i=0; i<3; i++){
                for(int j=0; j<3; j++){
                    if(board[i][j] == sign[k]){
                        input[glob] = 1.0;
                    }
                    else{
                        input[glob] = 0.0;
                    }
                    glob++;
                }
            }
        }

        //getting the network prediction
        float* prediction = xtpredict(kernel_ann, input);

        //get max
        int cell;
        float loc_max = FLT_MIN;
        for(int i=0;i<9;i++){
            if(*prediction > loc_max){
                loc_max = *prediction;
                cell = i;
            }
            prediction++;
        }
        
        // get the new mark
        int pos_x = cell%3;
        int pos_y = cell/3;
        // Move bestMove = findBestMove(board);
        // if(pos_y == bestMove.row && pos_x == bestMove.col) correct_choices++;

        //update board
        if(board[pos_y][pos_x] == '_'){
            board[pos_y][pos_x] = 'x';
        }
        else{
            break;
        }

        //random player
        int randomCell = randomPlay(board, id_random);
        int rand_pos_x = randomCell%3;
        int rand_pos_y = randomCell/3;
        board[rand_pos_y][rand_pos_x] = 'o';

        turns++;

        //next play
        // Move nextMove = findBestMove(board);
        // board[nextMove.row][nextMove.col] = 'o';
        
    }

    //calculate fitness
    ans resultado;
    resultado.id = id;
    resultado.fitness = turns;
    resultado.red = kernel_ann;
    respuestas[id] = resultado;
}