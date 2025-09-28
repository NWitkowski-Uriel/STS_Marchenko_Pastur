#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include "TFile.h"
#include "TTree.h"
#include "TMatrixD.h"
#include "TMatrixDSym.h"
#include "TVectorD.h"
#include "TH2D.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TF1.h"
#include "TLine.h"
#include "TLegend.h"
#include "TMath.h"
#include "TMatrixDSymEigen.h"
#include "CbmStsDigi.h"
#include "TSystem.h"
#include "TSystemDirectory.h"
#include "TList.h"
#include "TSystemFile.h"

using namespace std;

// Constants
const int MAX_TIME_BINS = 2000000;
const int SEGMENT_SIZE = 128;
const double MIN_BIN_WIDTH = 1.0;      // 1 microsecond
const double MAX_BIN_WIDTH = 5000.0;   // 5000 microseconds
const double DEFAULT_BIN_WIDTH = 1.0;  // Default bin width for analysis

// Structure to store analysis results
struct AnalysisResult {
    string fileName;
    string asicName;
    double binWidth;
    int numChannels;
    int numTimeBins;
    double ratioOutsideMP;
    int nearZeroEigenvalues;
    int eigenvaluesOutsideMP;
    int eigenvaluesAboveMP;
    int eigenvaluesBelowMP;
    double whiteningMaxDeviation;
    double whiteningAvgDeviation;
    double maxEigenvalue;
    double minEigenvalue;
};

// =====================================================================
// Marchenko-Pastur Distribution Function
// =====================================================================
Double_t MarchenkoPastur(Double_t* x, Double_t* par) {
    Double_t lambda = x[0];
    Double_t r = par[0];
    Double_t sigma2 = par[1];
    
    Double_t lambda_plus = sigma2 * TMath::Power(1 + TMath::Sqrt(r), 2);
    Double_t lambda_minus = sigma2 * TMath::Power(1 - TMath::Sqrt(r), 2);
    
    if (lambda < lambda_minus || lambda > lambda_plus) return 0;
    return TMath::Sqrt((lambda_plus - lambda) * (lambda - lambda_minus)) 
            / (2 * TMath::Pi() * sigma2 * lambda * r);
}

// =====================================================================
// Standardize data and compute covariance matrix
// =====================================================================
std::tuple<TMatrixDSym, TVectorD, TVectorD> ComputeCovarianceMatrix(const TMatrixD& data) {
    const Int_t N = data.GetNrows();
    const Int_t M = data.GetNcols();
    
    TVectorD means(N);
    TVectorD stddevs(N);
    TMatrixD standardized(N, M);

    for (Int_t i = 0; i < N; ++i) {
        Double_t sum = 0.0, sum2 = 0.0;
        for (Int_t t = 0; t < M; ++t) {
            sum += data(i, t);
            sum2 += data(i, t) * data(i, t);
        }
        means[i] = sum / M;
        Double_t variance = sum2/M - means[i]*means[i];
        
        // Handle zero-variance channels
        if (variance <= 0) {
            stddevs[i] = 1.0;
            for (Int_t t = 0; t < M; ++t) {
                standardized(i, t) = 0.0;
            }
        } else {
            stddevs[i] = TMath::Sqrt(variance);
            for (Int_t t = 0; t < M; ++t) {
                standardized(i, t) = (data(i,t) - means[i]) / stddevs[i];
            }
        }
    }

    // Compute covariance matrix
    TMatrixDSym covariance(N);
    for (Int_t i = 0; i < N; ++i) {
        for (Int_t j = 0; j <= i; ++j) {
            if (i == j) {
                covariance(i, j) = 1.0;
            } else {
                if (stddevs[i] == 1.0 && means[i] == 0.0 && 
                    stddevs[j] == 1.0 && means[j] == 0.0) {
                    covariance(i, j) = 0.0;
                } else if (stddevs[i] == 1.0 && means[i] == 0.0) {
                    covariance(i, j) = 0.0;
                } else if (stddevs[j] == 1.0 && means[j] == 0.0) {
                    covariance(i, j) = 0.0;
                } else {
                    Double_t cov = 0.0;
                    for (Int_t t = 0; t < M; ++t) {
                        cov += standardized(i, t) * standardized(j, t);
                    }
                    covariance(i, j) = cov / M;
                }
            }
        }
    }
    
    return std::make_tuple(covariance, means, stddevs);
}

// =====================================================================
// Verify Whitening Transformation
// =====================================================================
std::pair<double, double> VerifyWhitening(const TMatrixD& whiteningMatrix, const TMatrixDSym& covariance) {
    const Int_t N = covariance.GetNrows();
    
    // Calculate the transformed covariance: W * C * W^T should be identity
    TMatrixD W = whiteningMatrix;
    TMatrixD WT(whiteningMatrix);
    WT.T();
    
    TMatrixD temp1(N, N);
    temp1.Mult(covariance, WT);
    
    TMatrixD transformedCovariance(N, N);
    transformedCovariance.Mult(W,temp1);
    
    // Calculate deviation from identity
    Double_t maxDeviation = 0.0;
    Double_t avgDeviation = 0.0;
    Int_t count = 0;
    
    for (Int_t i = 0; i < N; ++i) {
        for (Int_t j = 0; j < N; ++j) {
            Double_t expected = (i == j) ? 1.0 : 0.0;
            Double_t deviation = TMath::Abs(transformedCovariance(i, j) - expected);
            avgDeviation += deviation;
            count++;
            if (deviation > maxDeviation) {
                maxDeviation = deviation;
            }
        }
    }
    avgDeviation /= count;
    
    return std::make_pair(maxDeviation, avgDeviation);
}

// =====================================================================
// Calculate Eigenspectrum with Whitening Matrix and MP statistics
// =====================================================================
std::tuple<TVectorD, TMatrixD, TMatrixD, TMatrixDSym, int, int, int, double, double> 
CalculateEigenspectrumWithMPStats(const TMatrixD& data) {
    const Int_t N = data.GetNrows();
    const Int_t M = data.GetNcols();
    if (N == 0 || M == 0) return std::make_tuple(TVectorD(), TMatrixD(), TMatrixD(), TMatrixDSym(), 0, 0, 0, 0.0, 0.0);

    // Compute covariance matrix
    auto [covariance, means, stddevs] = ComputeCovarianceMatrix(data);

    // Compute eigenvalues and eigenvectors
    TMatrixDSymEigen eigenCalc(covariance);
    TVectorD eigenvalues = eigenCalc.GetEigenValues();
    TMatrixD eigenvectors = eigenCalc.GetEigenVectors();

    // Calculate MP bounds
    Double_t r = (Double_t)N/M;
    Double_t lambda_plus = TMath::Power(1 + TMath::Sqrt(r), 2);
    Double_t lambda_minus = TMath::Power(1 - TMath::Sqrt(r), 2);

    // Count eigenvalues outside MP bounds
    int countOutside = 0;
    int countAbove = 0;
    int countBelow = 0;
    Double_t epsilon = 1e-10;
    
    for (Int_t i = 0; i < eigenvalues.GetNrows(); i++) {
        if (eigenvalues[i] > epsilon) {
            if (eigenvalues[i] > lambda_plus) {
                countOutside++;
                countAbove++;
            } else if (eigenvalues[i] < lambda_minus) {
                countOutside++;
                countBelow++;
            }
        }
    }

    // Create whitening matrix: W = Λ^(-1/2) * V^T
    TMatrixD whiteningMatrix(N, N);
    whiteningMatrix.Zero();
    
    TMatrixD invSqrtEigenvalueMatrix(N, N);
    invSqrtEigenvalueMatrix.Zero();
    
    for (Int_t i = 0; i < N; i++) {
        if (eigenvalues[i] > epsilon) {
            invSqrtEigenvalueMatrix(i, i) = 1.0 / TMath::Sqrt(eigenvalues[i]);
        } else {
            invSqrtEigenvalueMatrix(i, i) = 1.0;
        }
    }
    
    TMatrixD eigenvectorsT(eigenvectors);
    eigenvectorsT.T();
    whiteningMatrix.Mult(eigenvectorsT, invSqrtEigenvalueMatrix);

    // Verify whitening
    auto [maxDeviation, avgDeviation] = VerifyWhitening(whiteningMatrix, covariance);

    return std::make_tuple(eigenvalues, eigenvectors, whiteningMatrix, covariance, 
                          countOutside, countAbove, countBelow, maxDeviation, avgDeviation);
}

// =====================================================================
// Plot Eigenspectrum with Marchenko-Pastur Distribution
// =====================================================================
void PlotEigenspectrum(const TVectorD& eigenvalues, Int_t N, Int_t M, 
                      const string& asicName, double binWidth, const string& outputDir,
                      int eigenvaluesOutsideMP, int eigenvaluesAboveMP, int eigenvaluesBelowMP) {
    Double_t r = (Double_t)N/M;
    Double_t lambda_plus = TMath::Power(1 + TMath::Sqrt(r), 2);
    Double_t lambda_minus = TMath::Power(1 - TMath::Sqrt(r), 2);

    // Count near-zero eigenvalues
    Int_t nearZeroCount = 0;
    Double_t epsilon = 1e-3;
    Double_t max_eigenvalue = 0.0;
    Double_t min_eigenvalue = 1e10;
    
    for (Int_t i = 0; i < eigenvalues.GetNrows(); i++) {
        if (eigenvalues[i] < epsilon) {
            nearZeroCount++;
        }
        if (eigenvalues[i] > epsilon && eigenvalues[i] > max_eigenvalue) {
            max_eigenvalue = eigenvalues[i];
        }
        if (eigenvalues[i] > epsilon && eigenvalues[i] < min_eigenvalue) {
            min_eigenvalue = eigenvalues[i];
        }
    }
    
    Double_t x_max = TMath::Max(1.2 * max_eigenvalue, 4.0);
    x_max = TMath::Max(x_max, lambda_plus * 1.1);
    
    TH1D* hist = new TH1D(Form("eigen_hist_%s_bw%.0fus", asicName.c_str(), binWidth),
                         Form("Eigenvalue Distribution - %s (N=%d, M=%d, BW=%.0f#mus);#lambda;Density", 
                              asicName.c_str(), N, M, binWidth),
                         100, 0, x_max);
    
    for (Int_t i = 0; i < eigenvalues.GetNrows(); i++) {
        if (eigenvalues[i] > epsilon) {
            hist->Fill(eigenvalues[i]);
        }
    }
    
    if (hist->Integral() > 0) {
        hist->Scale(1.0 / hist->Integral("width"));
    }

    TF1* mpFunc = new TF1(Form("mpFunc_%s_bw%.0fus", asicName.c_str(), binWidth), 
                         MarchenkoPastur, 0, lambda_plus * 1.2, 2);
    mpFunc->SetParameters(r, 1.0);
    mpFunc->SetLineColor(kRed);
    mpFunc->SetLineWidth(2);
    mpFunc->SetLineStyle(2);

    TCanvas* canvas = new TCanvas(Form("eigen_canvas_%s_bw%.0fus", asicName.c_str(), binWidth),
                                 Form("Eigenspectrum - %s (BW=%.0f#mus)", asicName.c_str(), binWidth), 
                                 800, 600);
    
    gStyle->SetOptStat(0);
    gPad->SetGrid(1, 1);
    
    hist->SetLineColor(kBlue);
    hist->SetLineWidth(2);
    hist->SetFillColor(kBlue);
    hist->SetFillStyle(3004);
    hist->Draw("HIST");
    mpFunc->Draw("SAME");
    
    TLine* line_plus = new TLine(lambda_plus, 0, lambda_plus, hist->GetMaximum() * 1.1);
    line_plus->SetLineColor(kRed);
    line_plus->SetLineWidth(2);
    line_plus->SetLineStyle(2);
    line_plus->Draw();
    
    TLine* line_minus = new TLine(lambda_minus, 0, lambda_minus, hist->GetMaximum() * 1.1);
    line_minus->SetLineColor(kRed);
    line_minus->SetLineWidth(2);
    line_minus->SetLineStyle(2);
    line_minus->Draw();

    TLegend* legend = new TLegend(0.55, 0.65, 0.85, 0.85);
    legend->SetBorderSize(0);
    legend->SetFillStyle(0);
    legend->AddEntry(hist, "Eigenvalues", "lf");
    legend->AddEntry(mpFunc, "Marchenko-Pastur", "l");
    legend->AddEntry(line_plus, Form("#lambda_{+} = %.3f", lambda_plus), "l");
    legend->AddEntry(line_minus, Form("#lambda_{-} = %.3f", lambda_minus), "l");
    legend->Draw();

    // Add statistics below the legend - only the requested information
    TLatex* latex = new TLatex();
    latex->SetNDC();
    latex->SetTextSize(0.03);
    latex->SetTextAlign(12);
    
    // Position statistics below the legend
    latex->DrawLatex(0.55, 0.60, Form("Outside MP: %d", eigenvaluesOutsideMP));
    latex->DrawLatex(0.55, 0.56, Form("Above #lambda_{+}: %d", eigenvaluesAboveMP));
    latex->DrawLatex(0.55, 0.52, Form("Below #lambda_{-}: %d", eigenvaluesBelowMP));
    latex->DrawLatex(0.55, 0.48, Form("Near-zero: %d", nearZeroCount));

    // Save to file
    canvas->SaveAs(Form("%s/eigenspectrum_%s_bw%.0fus.png", outputDir.c_str(), asicName.c_str(), binWidth));

    delete canvas;
    delete hist;
    delete mpFunc;
    delete line_plus;
    delete line_minus;
    delete legend;
    delete latex;
}

// =====================================================================
// Draw Covariance Matrix Heatmap
// =====================================================================
void DrawCovarianceHeatMap(const TMatrixDSym& matrix, const string& asicName, 
                           double binWidth, Int_t N, Int_t M, const string& outputDir) {
    TH2D* hist = new TH2D(Form("covariance_heatmap_%s_bw%.0fus", asicName.c_str(), binWidth),
        Form("Covariance Matrix - %s (N=%d, M=%d, BW=%.0f#mus);Channel Index;Channel Index", 
             asicName.c_str(), N, M, binWidth),
        N, 0, N,
        N, 0, N);
    
    for (Int_t i = 0; i < N; ++i) {
        for (Int_t j = 0; j < N; ++j) {
            if(j <= i){
                hist->SetBinContent(i+1, N - j, matrix(i,j));
            } else {
                hist->SetBinContent(i+1, N - j, matrix(j,i));
            }           
        }
    }

    hist->GetXaxis()->CenterTitle();
    hist->GetYaxis()->CenterTitle();
    hist->GetYaxis()->SetRangeUser(N, 0);
    
    hist->GetXaxis()->SetBinLabel(1, "1");
    hist->GetXaxis()->SetBinLabel(N, Form("%d", N));
    hist->GetYaxis()->SetBinLabel(1, Form("%d", N));
    hist->GetYaxis()->SetBinLabel(N, "1");

    TCanvas* canvas = new TCanvas(Form("covariance_heatmap_%s_bw%.0fus", asicName.c_str(), binWidth),
        Form("Covariance Matrix - %s (BW=%.0f#mus)", asicName.c_str(), binWidth), 1000, 800);
    canvas->SetRightMargin(0.15);
    canvas->SetLeftMargin(0.12);
    canvas->SetBottomMargin(0.12);
    
    gStyle->SetOptStat(0);
    gStyle->SetPalette(kRainBow);
    hist->SetContour(100);
    hist->GetZaxis()->SetTitle("Covariance");
    hist->SetMinimum(-1.0);
    hist->SetMaximum(1.0);
    hist->Draw("COLZ");

    canvas->SaveAs(Form("%s/covariance_heatmap_%s_bw%.0fus.png", outputDir.c_str(), asicName.c_str(), binWidth));

    delete canvas;
    delete hist;
}

// =====================================================================
// Draw Whitening Matrix Heatmap
// =====================================================================
void DrawWhiteningHeatMap(const TMatrixD& matrix, const string& asicName, 
                         double binWidth, Int_t N, Int_t M, const string& outputDir) {
    TH2D* hist = new TH2D(Form("whitening_heatmap_%s_bw%.0fus", asicName.c_str(), binWidth),
        Form("Whitening Matrix - %s (N=%d, M=%d, BW=%.0f#mus);Output Dimension;Input Dimension", 
             asicName.c_str(), N, M, binWidth),
        N, 0, N,
        N, 0, N);
    
    // Find min and max values for better color scaling
    Double_t minVal = 1e10;
    Double_t maxVal = -1e10;
    for (Int_t i = 0; i < N; ++i) {
        for (Int_t j = 0; j < N; ++j) {
            Double_t val = matrix(i, j);
            if (val < minVal) minVal = val;
            if (val > maxVal) maxVal = val;
        }
    }
    
    // Fill the histogram
    for (Int_t i = 0; i < N; ++i) {
        for (Int_t j = 0; j < N; ++j) {
            hist->SetBinContent(i+1, N - j, matrix(i, j));
        }
    }

    hist->GetXaxis()->CenterTitle();
    hist->GetYaxis()->CenterTitle();
    hist->GetYaxis()->SetRangeUser(N, 0);
    
    hist->GetXaxis()->SetBinLabel(1, "1");
    hist->GetXaxis()->SetBinLabel(N, Form("%d", N));
    hist->GetYaxis()->SetBinLabel(1, Form("%d", N));
    hist->GetYaxis()->SetBinLabel(N, "1");

    TCanvas* canvas = new TCanvas(Form("whitening_heatmap_%s_bw%.0fus", asicName.c_str(), binWidth),
        Form("Whitening Matrix - %s (BW=%.0f#mus)", asicName.c_str(), binWidth), 1000, 800);
    canvas->SetRightMargin(0.15);
    canvas->SetLeftMargin(0.12);
    canvas->SetBottomMargin(0.12);
    
    gStyle->SetOptStat(0);
    gStyle->SetPalette(kRainBow);
    hist->SetContour(100);
    hist->GetZaxis()->SetTitle("Whitening Coefficient");
    
    // Use symmetric scaling for better visualization
    Double_t absMax = TMath::Max(TMath::Abs(minVal), TMath::Abs(maxVal));
    hist->SetMinimum(-absMax);
    hist->SetMaximum(absMax);
    
    hist->Draw("COLZ");

    canvas->SaveAs(Form("%s/whitening_heatmap_%s_bw%.0fus.png", outputDir.c_str(), asicName.c_str(), binWidth));

    delete canvas;
    delete hist;
}

// =====================================================================
// Convert nanoseconds to microseconds
// =====================================================================
double nsToUs(double nanoseconds) {
    return nanoseconds / 1000.0;
}

// =====================================================================
// Extract base file name without path and extension
// =====================================================================
string GetBaseFileName(const string& fullPath) {
    string baseName = fullPath;
    size_t lastSlash = baseName.find_last_of("/\\");
    if (lastSlash != string::npos) {
        baseName = baseName.substr(lastSlash + 1);
    }
    size_t dotPos = baseName.find_last_of(".");
    if (dotPos != string::npos) {
        baseName = baseName.substr(0, dotPos);
    }
    return baseName;
}

// =====================================================================
// Find all .digi.root files recursively
// =====================================================================
vector<string> FindDigiRootFiles(const string& directory) {
    vector<string> files;
    
    TSystemDirectory dir(directory.c_str(), directory.c_str());
    TList* fileList = dir.GetListOfFiles();
    
    if (fileList) {
        TIter next(fileList);
        TSystemFile* file;
        while ((file = (TSystemFile*)next())) {
            TString fileName = file->GetName();
            TString filePath = directory + "/" + fileName;
            
            if (file->IsDirectory()) {
                if (fileName != "." && fileName != "..") {
                    // Recursively search subdirectories
                    vector<string> subFiles = FindDigiRootFiles(filePath.Data());
                    files.insert(files.end(), subFiles.begin(), subFiles.end());
                }
            } else if (fileName.EndsWith(".digi.root")) {
                files.push_back(filePath.Data());
            }
        }
        delete fileList;
    }
    
    return files;
}

// =====================================================================
// Analyze a single file
// =====================================================================
vector<AnalysisResult> AnalyzeFile(const string& inputFile, double binWidthUs, const string& baseOutputDir) {
    vector<AnalysisResult> results;
    
    // Create file-specific output directory
    string baseFileName = GetBaseFileName(inputFile);
    string fileOutputDir = baseOutputDir + "/" + baseFileName;
    gSystem->mkdir(fileOutputDir.c_str(), kTRUE);
    
    cout << "Output directory for file: " << fileOutputDir << endl;
    
    TFile* file = TFile::Open(inputFile.c_str());
    if (!file || file->IsZombie()) {
        cerr << "Error: Could not open input file: " << inputFile << endl;
        return results;
    }

    TTree* tree = (TTree*)file->Get("cbmsim");
    if (!tree) {
        cerr << "Error: Could not find tree cbmsim in file: " << inputFile << endl;
        file->Close();
        return results;
    }

    vector<CbmStsDigi>* digiVector = nullptr;
    tree->SetBranchAddress("StsDigi", &digiVector);

    cout << "Analyzing file: " << inputFile << endl;

    // Map to store timestamps for each ASIC
    map<string, set<double>> asicTimestamps;
    map<int, string> channelToAsic;
    
    Long64_t nEntries = tree->GetEntries();
    
    // First pass: collect timestamps and map channels to ASICs
    for (Long64_t iEntry = 0; iEntry < nEntries; iEntry++) {
        tree->GetEntry(iEntry);
        for (const auto& digi : *digiVector) {
            int channel = digi.GetChannel();
            double time = digi.GetTime(); // time in nanoseconds
            
            // Determine ASIC name based on channel
            int asicIndex = channel / SEGMENT_SIZE;
            string asicName;
            if (asicIndex < 8) {
                asicName = Form("N%d", asicIndex);
            } else if (asicIndex < 16) {
                asicName = Form("P%d", asicIndex - 8);
            } else {
                asicName = Form("Extra%d", asicIndex);
            }
            
            channelToAsic[channel] = asicName;
            asicTimestamps[asicName].insert(time);
        }
    }

    // Convert bin width from microseconds to nanoseconds for internal calculations
    double binWidthNs = binWidthUs * 1000.0;
    
    // Process each ASIC separately
    for (const auto& asicEntry : asicTimestamps) {
        const string& asicName = asicEntry.first;
        const set<double>& asicTimes = asicEntry.second;
        
        if (asicTimes.empty()) continue;

        double minTime = *asicTimes.begin(); // in nanoseconds
        double maxTime = *asicTimes.rbegin(); // in nanoseconds
        double totalTimeRangeNs = maxTime - minTime;
        
        // Calculate number of time bins for this bin width
        int numTimeBins = static_cast<int>(totalTimeRangeNs / binWidthNs) + 1;
        
        // Enforce column limit
        if (numTimeBins > MAX_TIME_BINS) {
            cout << "WARNING: ASIC " << asicName << " would require " << numTimeBins 
                 << " time bins, truncating to " << MAX_TIME_BINS << " bins." << endl;
            numTimeBins = MAX_TIME_BINS;
        }
        
        if (numTimeBins < 2) numTimeBins = 2; // Minimum 2 bins
        
        cout << "  ASIC " << asicName << ": " << numTimeBins << " time bins of " << binWidthUs << "μs" << endl;

        // Get ALL channels for this ASIC (0-127)
        vector<int> asicChannels;
        int asicIndex;
        if (asicName[0] == 'N') {
            asicIndex = std::stoi(asicName.substr(1));
        } else if (asicName[0] == 'P') {
            asicIndex = std::stoi(asicName.substr(1)) + 8;
        } else {
            asicIndex = std::stoi(asicName.substr(5));
        }
        
        int startChannel = asicIndex * SEGMENT_SIZE;
        for (int i = 0; i < SEGMENT_SIZE; i++) {
            asicChannels.push_back(startChannel + i);
        }
        
        const Int_t N = asicChannels.size();

        TMatrixD asicData(N, numTimeBins);
        asicData.Zero();

        // Create mapping from channel number to row index
        map<int, int> channelToRowIndex;
        for (int i = 0; i < N; i++) {
            channelToRowIndex[asicChannels[i]] = i;
        }

        // Fill the data matrix with time window binning
        for (Long64_t iEntry = 0; iEntry < nEntries; iEntry++) {
            tree->GetEntry(iEntry);
            for (const auto& digi : *digiVector) {
                int channel = digi.GetChannel();
                
                if (channelToAsic[channel] != asicName) continue;
                
                double time = digi.GetTime();
                int charge = digi.GetCharge();
                
                int binIdx = static_cast<int>((time - minTime) / binWidthNs);
                
                if (binIdx < 0 || binIdx >= numTimeBins) continue;
                
                auto it = channelToRowIndex.find(channel);
                if (it != channelToRowIndex.end()) {
                    int channelIdx = it->second;
                    asicData(channelIdx, binIdx) += charge;
                }
            }
        }

        // Compute eigenvalue matrix, eigenvector matrix, and whitening matrix with MP statistics
        auto result = CalculateEigenspectrumWithMPStats(asicData);
        TVectorD eigenvalues = std::get<0>(result);
        TMatrixD eigenvectors = std::get<1>(result);
        TMatrixD whiteningMatrix = std::get<2>(result);
        TMatrixDSym covariance = std::get<3>(result);
        int eigenvaluesOutsideMP = std::get<4>(result);
        int eigenvaluesAboveMP = std::get<5>(result);
        int eigenvaluesBelowMP = std::get<6>(result);
        double whiteningMaxDeviation = std::get<7>(result);
        double whiteningAvgDeviation = std::get<8>(result);

        // Calculate additional statistics
        Double_t epsilon = 1e-10;
        int nearZeroCount = 0;
        double maxEigenvalue = 0.0;
        double minEigenvalue = 1e10;
        
        for (Int_t i = 0; i < eigenvalues.GetNrows(); i++) {
            if (eigenvalues[i] < epsilon) {
                nearZeroCount++;
            } else {
                if (eigenvalues[i] > maxEigenvalue) maxEigenvalue = eigenvalues[i];
                if (eigenvalues[i] < minEigenvalue) minEigenvalue = eigenvalues[i];
            }
        }
        
        double ratioOutside = (double)eigenvaluesOutsideMP / N;

        // Store result
        AnalysisResult res;
        res.fileName = inputFile;
        res.asicName = asicName;
        res.binWidth = binWidthUs;
        res.numChannels = N;
        res.numTimeBins = numTimeBins;
        res.ratioOutsideMP = ratioOutside;
        res.nearZeroEigenvalues = nearZeroCount;
        res.eigenvaluesOutsideMP = eigenvaluesOutsideMP;
        res.eigenvaluesAboveMP = eigenvaluesAboveMP;
        res.eigenvaluesBelowMP = eigenvaluesBelowMP;
        res.whiteningMaxDeviation = whiteningMaxDeviation;
        res.whiteningAvgDeviation = whiteningAvgDeviation;
        res.maxEigenvalue = maxEigenvalue;
        res.minEigenvalue = minEigenvalue;
        results.push_back(res);

        // Generate plots
        PlotEigenspectrum(eigenvalues, N, numTimeBins, asicName, binWidthUs, fileOutputDir,
                 eigenvaluesOutsideMP, eigenvaluesAboveMP, eigenvaluesBelowMP);
        DrawCovarianceHeatMap(covariance, asicName, binWidthUs, N, numTimeBins, fileOutputDir);
        DrawWhiteningHeatMap(whiteningMatrix, asicName, binWidthUs, N, numTimeBins, fileOutputDir);

        // Save matrices to ROOT file
        TFile* matrixFile = new TFile(Form("%s/matrices_%s_bw%.0fus.root", fileOutputDir.c_str(), asicName.c_str(), binWidthUs), "RECREATE");
        eigenvalues.Write("eigenvalues");
        eigenvectors.Write("eigenvectors");
        whiteningMatrix.Write("whiteningMatrix");
        covariance.Write("covarianceMatrix");
        
        // Save MP statistics
        TVectorD mpStats(5);
        mpStats[0] = eigenvaluesOutsideMP;
        mpStats[1] = eigenvaluesAboveMP;
        mpStats[2] = eigenvaluesBelowMP;
        mpStats[3] = whiteningMaxDeviation;
        mpStats[4] = whiteningAvgDeviation;
        mpStats.Write("mpStatistics");
        
        matrixFile->Close();
        
        cout << "    Saved matrices and plots for ASIC " << asicName << " in " << fileOutputDir << endl;
        cout << "    Eigenvalues outside MP: " << eigenvaluesOutsideMP << " (" << ratioOutside*100 << "%)" << endl;
        cout << "    Whitening verification - Max: " << whiteningMaxDeviation << ", Avg: " << whiteningAvgDeviation << endl;
    }

    file->Close();
    return results;
}

// =====================================================================
// Main Analysis Function
// =====================================================================
void AnalyzeAllDigiRootFiles(const string& searchDirectory = ".", 
                           double binWidthUs = DEFAULT_BIN_WIDTH) {
    // Create main output directory
    string mainOutputDir = Form("MP_Analysis_Results_bw%.0fus", binWidthUs);
    gSystem->mkdir(mainOutputDir.c_str(), kTRUE);
    
    cout << "Searching for .digi.root files in: " << searchDirectory << endl;
    
    // Find all .digi.root files
    vector<string> digiFiles = FindDigiRootFiles(searchDirectory);
    
    if (digiFiles.empty()) {
        cout << "No .digi.root files found in " << searchDirectory << endl;
        return;
    }
    
    cout << "Found " << digiFiles.size() << " .digi.root files:" << endl;
    for (const auto& file : digiFiles) {
        cout << "  " << file << endl;
    }
    
    vector<AnalysisResult> allResults;
    
    // Analyze each file
    for (const auto& file : digiFiles) {
        vector<AnalysisResult> fileResults = AnalyzeFile(file, binWidthUs, mainOutputDir);
        allResults.insert(allResults.end(), fileResults.begin(), fileResults.end());
    }
    
    // Save summary results in the main output directory
    ofstream summaryFile(Form("%s/summary_results.csv", mainOutputDir.c_str()));
    summaryFile << "File,ASIC,BinWidth(us),Channels,TimeBins,RatioOutsideMP,NearZeroEigenvalues,"
                << "EigenvaluesOutsideMP,EigenvaluesAboveMP,EigenvaluesBelowMP,"
                << "WhiteningMaxDeviation,WhiteningAvgDeviation,MaxEigenvalue,MinEigenvalue" << endl;
    
    for (const auto& res : allResults) {
        summaryFile << res.fileName << "," << res.asicName << "," << res.binWidth << ","
                   << res.numChannels << "," << res.numTimeBins << "," << res.ratioOutsideMP << ","
                   << res.nearZeroEigenvalues << "," << res.eigenvaluesOutsideMP << ","
                   << res.eigenvaluesAboveMP << "," << res.eigenvaluesBelowMP << ","
                   << res.whiteningMaxDeviation << "," << res.whiteningAvgDeviation << ","
                   << res.maxEigenvalue << "," << res.minEigenvalue << endl;
    }
    summaryFile.close();
    
    cout << "\nAnalysis complete! Results saved in: " << mainOutputDir << endl;
    cout << "Summary file: " << mainOutputDir << "/summary_results.csv" << endl;
    cout << "Each input file has its own subdirectory with detailed results" << endl;
    cout << "Key metrics included:" << endl;
    cout << "  - Number of eigenvalues outside MP spectrum" << endl;
    cout << "  - Number above/below MP bounds" << endl;
    cout << "  - Whitening matrix verification results" << endl;
    cout << "  - Maximum and minimum eigenvalues" << endl;
}

// =====================================================================
// Main function
// =====================================================================
void run_mp_analysis() {
    // Analyze all .digi.root files with default bin width
    AnalyzeAllDigiRootFiles(".", DEFAULT_BIN_WIDTH);
}