#include <bits/stdc++.h>
#include <windows.h>   // WinApi header
#include <conio.h>
using namespace std;    // std::cout, std::cin

int main()
{
    HANDLE  hConsole;
    int k;
    ifstream in;
    in.open("test_x.txt");
    hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    size_t n;
    char c;
    while(true)
    {
        in.get(c);
        if(c == '\n')
        {
            cout << '\n';
            cin.get();
            system("cls");
            continue;
        }
        while(true)
        {
            n = 0;
            if(c == '\n')
            {
                cout << '\n';
                break;
            }
            while(c != ' ')
            {
                n *= 10;
                n += c-'0';
                in.get(c);
            }
            if(n != 0)
            {
                SetConsoleTextAttribute(hConsole, n);
                cout << ' ';
                SetConsoleTextAttribute(hConsole, 0);
            }else
            {
                SetConsoleTextAttribute(hConsole, 0);
                cout << ' ';
            }
            in.get(c);
        }
    }
    return 0;
}
