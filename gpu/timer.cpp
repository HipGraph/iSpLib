#include <iostream>

long cuda_spmm_test(int);

int main()
{
    long long lowest = 99999999;
    int lowest_i = 0;

    for (int i = 1; i <= 1024; i++)
    {
        auto t = cuda_spmm_test(i);
        if (t < lowest){
            lowest = t;
            lowest_i = i;
        }
        std::cout << i << "\t" << t << "\n";
        // break;
    }
    if (lowest_i != 0){
        std::cout << lowest_i << " is lowest, time: \t" << lowest << "\n";
    }
    return 0;
}
