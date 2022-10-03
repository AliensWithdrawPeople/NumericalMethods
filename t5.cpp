#include<iostream>
#include<TGraph.h>
#include<vector>
#include<tuple>
#include<TGraph.h>
#include<TMath.h>

class LagrangePol
{
private:
    std::vector<double> w;
    std::vector<double> dataX;
    std::vector<double> dataY;
public:
    LagrangePol(const std::vector<double> &datX, const std::vector<double> &datY);
    double Eval(double x);
    ~LagrangePol();
};

LagrangePol::LagrangePol(const std::vector<double> &datX, const std::vector<double> &datY)
{
    dataX = datX;
    dataY = datY;
    double tmp = 1;
    for(int i = 0; i < dataX.size(); i++)
    {
        for(int j = 0; j < dataX.size(); j++)
        {
            if(i != j)
            { tmp *= 1 / (dataX[i] - dataX[j]); }
        }
        w.push_back(tmp);
        tmp = 1;
    }
}

double LagrangePol::Eval(double x)
{
    double sum1 = 0;
    double sum2 = 0;
    for(int i = 0; i < dataX.size(); i++)
    {
        if(fabs(x - dataX[i]) < 1e-8)
        { return dataY[i]; }
        sum1 += w[i] * dataY[i] / (x - dataX[i]);
        sum2 += w[i] / (x - dataX[i]);
    }
    return sum1 / sum2;
}

LagrangePol::~LagrangePol()
{
    w.clear();
    dataX.clear();
    dataY.clear();
}

class NewtonPol
{
private:
    std::vector<std::vector<double>> coff;
    std::vector<double> dataX;
    std::vector<double> dataY;
public:
    NewtonPol(const std::vector<double> &datX, const std::vector<double> &datY);
    double Eval(double x);
    ~NewtonPol();
};

NewtonPol::NewtonPol(const std::vector<double> &datX, const std::vector<double> &datY)
{
    dataX = datX;
    dataY = datY;
    for(int i = 0; i < dataY.size(); i++)
    { coff.push_back({dataY[i]}); }
    for (int i = 1; i < dataY.size(); i++) 
    {
        for (int j = 0; j < dataY.size() - i; j++) 
        { coff[j].push_back( (coff[j][i - 1] - coff[j + 1][i - 1]) / (dataX[j] - dataX[i + j]) ); }
    }
}

NewtonPol::~NewtonPol()
{
    dataX.clear();
    dataY.clear();
    coff.clear();
}

double NewtonPol::Eval(double x)
{
    double val = 0;
    double prod = 1;
    for (int i = 0; i < dataX.size(); i++)
    {
        for (int j = 0; j < i; j++)
        { prod *= x - dataX[j]; }
        val += coff[0][i] * prod;
        prod = 1;
    }
    return val;
}


int t5()
{
    int n = 16;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> xAn;
    std::vector<double> yAn;

    std::vector<double> x2;
    std::vector<double> y2;
    std::vector<double> y3;

    double xVal = 0;
    for(int i = 0; i < n; i++)
    { 
        // xVal = -5. + i * 10. / n;
        xVal = 5 * cos((2 * (i + 1) - 1) * TMath::Pi() / 2 / n);
        x.push_back(xVal); 
        y.push_back(1 / (1 + xVal * xVal) ); 
    }
    auto pol = new LagrangePol(x, y);
    auto pol2 = new NewtonPol(x, y);
    int pointsNum = 100;
    for(int i = 0; i < pointsNum; i++)
    {
        // xVal = -5 + i * 10. / pointsNum;
        xVal = 5 * cos((2 * (i + 1) - 1) * TMath::Pi() / 2 / pointsNum);
        yAn.push_back(1 / (1 + xVal * xVal));
        std::cout << 1 / (1 + xVal * xVal) << std::endl;
        x2.push_back(xVal);
        y2.push_back(pol->Eval(xVal));
        // y2.push_back(pol->Eval(xVal) - 1 / (1 + xVal * xVal));

        y3.push_back(pol2->Eval(xVal));
        // std::cout << pol2->Eval(xVal) << std::endl;
        // y3.push_back(pol2->Eval(xVal) - 1 / (1 + xVal * xVal));
    }
    TGraph gr(x2.size(), x2.data(), yAn.data());
    TGraph grPol(x2.size(), x2.data(), y2.data());
    TGraph grPol2(x2.size(), x2.data(), y3.data());
    grPol2.SetLineColor(kBlue);
    grPol.SetLineColor(kGreen);
    // grPol.SetMarkerSize(1);
    grPol2.SetMarkerColor(kRed);
    std::cout << x2.size() << std::endl;
    gr.DrawClone();
    grPol.DrawClone("same");
    grPol2.DrawClone("same");
    return 0;
}