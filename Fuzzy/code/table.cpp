#include<iostream>
#include<bits/stdc++.h>
using namespace std;

vector<vector<string> > csv_table(string& file_name) {
    ifstream file(file_name);
    vector<vector<string> > table;
    string line;
    while (getline(file, line)) {
        vector<string> row;
        string cell;
        for (int i = 0; i < line.size(); i++) {
            if (line[i] == ',') {
                row.push_back(cell);
                cell = "";
            }
            else {
                cell += line[i];
            }
        }
        row.push_back(cell);
        table.push_back(row);
    }
    return table;
}


int main() {
    string file_name = "register_2.csv";
    auto matrix = csv_table(file_name);
    int no_fields = matrix[0].size();
    cout<<"\\begin{tabular}{|";
    for(int i=0;i<no_fields;i++) {
        cout<<"c|";
    }
    cout<<"}"<<endl;
    cout<<"\\hline\n";
    for(int i=0;i<matrix.size();i++) {
        cout<<matrix[i][0];
        for(int j=1;j<matrix[i].size();j++) {
            if(j == 5) continue;
            cout<<" & "<<matrix[i][j];
        }
        cout<<"\\\\"<<endl;
        cout<<"\\hline"<<endl;
    }
    cout<<"\\end{tabular}\n";
}