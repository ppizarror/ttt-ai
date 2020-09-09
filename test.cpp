#include "Tinn.h"
#include <bits/stdc++.h>
using namespace std;

const char* path = "mejor_red";

float* generate_input(char board[3][3]);
int jugada_ai(Tinn ai, float* input);
bool isMovesLeft(char board[3][3]);
void print_board(char board[3][3]);

int main()
{
    Tinn ttt_ai = xtload(path);
    char board[3][3];

    for(int i=0;i<3;i++){
        for(int j=0;j<3; j++){
            board[i][j] = '_';
        }
    }

    cout << "Tablero: " << endl;
    print_board(board);

    bool first_turn = true;

    while(isMovesLeft(board))
    {   
        if(!first_turn)
        {
            int x,y;
            cout << "Jugada: " << endl;
            cin >> x >> y;
            board[x][y] = 'x';
            print_board(board);
        }
        first_turn = false;
        float* input = generate_input(board);
        int cell = jugada_ai(ttt_ai, input);
        int pos_x = cell%3;
        int pos_y = cell/3;
        if(board[pos_y][pos_x] == '_'){
            board[pos_y][pos_x] = 'o';
        }
        else{
            cout << "La red jugó en una posicion no válida" << endl;
            break;
        }
        cout << "Juega la red: " << endl;
        print_board(board);

    }
    cout << "Tablero final: " << endl;
    print_board(board);
    return 0;
}

void print_board(char board[3][3])
{
    cout << endl;
    for(int i=0;i<3;i++){
        for(int j=0;j<3; j++){
            cout << board[i][j] << " " ;
        }
        cout << endl;
    }
    cout << endl;
}


bool isMovesLeft(char board[3][3]) 
{ 
    for (int i = 0; i<3; i++) 
        for (int j = 0; j<3; j++) 
            if (board[i][j]=='_') 
                return true;
    return false; 
}

int jugada_ai(Tinn ai, float* input)
{
    //getting the network prediction
    float* prediction = xtpredict(ai, input);

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

    return cell;
}


float* generate_input(char board[3][3])
{
    float* input = (float*) calloc(27, sizeof(float));
    char sign[3] = {'x', 'o', '_'};
    int glob = 0;
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

    return input;
}
