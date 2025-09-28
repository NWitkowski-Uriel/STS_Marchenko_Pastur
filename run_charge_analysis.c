#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TH3F.h>
#include <TChain.h>
#include <TSystem.h>
#include <TString.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TLegend.h>
#include <TPaveText.h>
#include <TGraph2D.h>
#include <TGraph.h>
#include <TF1.h>
#include <TProfile.h>
#include <TLatex.h>
#include <iostream>
#include <vector>
#include <map>
#include <TDirectory.h>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <TMath.h>
#include <memory>

// Forward declarations
class CbmStsDigi;

// Structure to hold calibration data
struct CalibrationData {
    Int_t runNumber;
    Int_t Vreft;
    Int_t VrefT_z_strips;
    Int_t cas_csa;
    TString calibration_params;
    Int_t cas_shap;
    TString comments;
    
    CalibrationData() : runNumber(0), Vreft(-999), VrefT_z_strips(-999), 
                       cas_csa(-999), cas_shap(-999) {}
};

// Time interval data structure
struct TimeIntervalData {
    std::vector<double> global_intervals;
    std::vector<double> nside_intervals;
    std::vector<double> pside_intervals;
    std::map<int, std::vector<double>> channel_intervals;
    
    void clear() {
        global_intervals.clear();
        nside_intervals.clear();
        pside_intervals.clear();
        channel_intervals.clear();
    }
};

// Analysis results structure
struct AnalysisResult {
    Int_t runNumber;
    Double_t mean_charge;
    Double_t rms_charge;
    Double_t mean_charge_flux;
    Double_t mean_charge_flux_nside;
    Double_t mean_charge_flux_pside;
    Double_t rms_charge_flux;
    Int_t total_hits;
    Int_t active_channels;
    Int_t active_channels_nside;
    Int_t active_channels_pside;
    Int_t Vreft;
    Int_t VrefT_z_strips;
    Int_t cas_csa;
    TString calibration_params;
    Int_t cas_shap;
    TString comments;
    
    AnalysisResult() : runNumber(0), mean_charge(0), rms_charge(0), mean_charge_flux(0),
                      mean_charge_flux_nside(0), mean_charge_flux_pside(0), rms_charge_flux(0),
                      total_hits(0), active_channels(0), active_channels_nside(0),
                      active_channels_pside(0), Vreft(-999), VrefT_z_strips(-999),
                      cas_csa(-999), cas_shap(-999) {}
};

// Global containers
std::vector<AnalysisResult> gAnalysisResults;
std::map<int, std::map<int, double>> gLastHitTime;
TimeIntervalData gTimeIntervals;

// Forward declarations
std::map<int, CalibrationData> read_calibration_data(const TString& filename);
int extract_run_number(const TString& filename);
void calculate_time_intervals(int runNumber, const std::vector<CbmStsDigi>& digiVector);
void analyze_single_file(const TString& inputFile, const TString& outputBaseDir, 
                        const std::map<int, CalibrationData>& calibrationMap);
void analyze_cross_run_time_intervals(const TString& outputDir);
void create_3d_histogram(const TString& outputDir);
void analyze_digi_charge();

// Helper function to create and save plots
template<typename Func>
void create_and_save_plot(const TString& plotsDir, const TString& name, Func drawFunc, int width = 800, int height = 600) {
    TCanvas canvas(name, name, width, height);
    drawFunc();
    canvas.SaveAs(Form("%s/%s.png", plotsDir.Data(), name.Data()));
}

// Function to read calibration data from file
std::map<int, CalibrationData> read_calibration_data(const TString& filename) {
    std::map<int, CalibrationData> calibrationMap;
    
    std::ifstream calibFile(filename.Data());
    if (!calibFile.is_open()) {
        std::cerr << "Error: Could not open calibration file: " << filename << std::endl;
        return calibrationMap;
    }

    std::string line;
    int lineNumber = 0;
    
    while (std::getline(calibFile, line)) {
        lineNumber++;
        
        if (line.empty() || line[0] == '#') {
            continue;
        }

        CalibrationData data;
        std::istringstream iss(line);
        
        if (!(iss >> data.runNumber)) {
            continue;
        }
        
        iss >> data.Vreft >> data.VrefT_z_strips >> data.cas_csa;
        
        std::string temp;
        if (iss >> temp) data.calibration_params = temp;
        iss >> data.cas_shap;
        
        calibrationMap[data.runNumber] = data;
    }
    calibFile.close();
    
    std::cout << "Successfully read " << calibrationMap.size() << " calibration entries" << std::endl;
    return calibrationMap;
}

// Function to extract run number from filename
int extract_run_number(const TString& filename) {
    TString baseName = gSystem->BaseName(filename);
    baseName.ReplaceAll(".digi.root", "");
    
    if (baseName.BeginsWith("run_")) {
        baseName.Replace(0, 4, "");
    }
    
    return atoi(baseName.Data());
}

// Function to calculate time intervals
void calculate_time_intervals(int runNumber, const std::vector<CbmStsDigi>& digiVector) {
    auto& lastHitTimes = gLastHitTime[runNumber];
    
    for (const auto& digi : digiVector) {
        int channel = digi.GetChannel();
        double time = digi.GetTime();
        
        if (lastHitTimes.find(channel) != lastHitTimes.end()) {
            double interval = time - lastHitTimes[channel];
            if (interval > 0) {
                gTimeIntervals.global_intervals.push_back(interval);
                
                if (channel < 1024) {
                    gTimeIntervals.nside_intervals.push_back(interval);
                } else {
                    gTimeIntervals.pside_intervals.push_back(interval);
                }
                
                gTimeIntervals.channel_intervals[channel].push_back(interval);
            }
        }
        lastHitTimes[channel] = time;
    }
}

// Function to create all plots
void create_all_plots(const TString& plotsDir, 
                     TH1F* hTotalCharge, TH1F* hMeanChargeFlux, TH1F* hChannelHits,
                     TH1F* hChannelHits_NSide, TH1F* hChannelHits_PSide,
                     TH1F* hChannelHitsLog, TH1F* hChannelHits_NSideLog, TH1F* hChannelHits_PSideLog,
                     TH1F* hMeanFluxDistribution, TH1F* hMeanFluxDistributionNSide, TH1F* hMeanFluxDistributionPSide,
                     TH1F* hMeanChargeFluxNSide, TH1F* hMeanChargeFluxPSide,
                     TH1F* hTimeIntervalGlobal, TH1F* hTimeIntervalNSide, TH1F* hTimeIntervalPSide, TH1F* hTimeIntervalLog) {
    
    // Individual plots
    create_and_save_plot(plotsDir, "charge_distribution", [&]() { 
        hTotalCharge->Draw(); 
    });
    
    create_and_save_plot(plotsDir, "mean_charge_flux", [&]() { 
        hMeanChargeFlux->Draw(); 
    });
    
    create_and_save_plot(plotsDir, "channel_hits", [&]() { 
        hChannelHits->Draw(); 
    });
    
    create_and_save_plot(plotsDir, "channel_hits_log", [&]() { 
        gPad->SetLogy();
        hChannelHitsLog->Draw(); 
    });
    
    // N-side vs P-side hits comparison
    create_and_save_plot(plotsDir, "nside_pside_hits", [&]() {
        TCanvas canvas("nside_pside_hits", "N-side vs P-side Hits", 1200, 800);
        canvas.Divide(2, 1);
        canvas.cd(1); hChannelHits_NSide->Draw();
        canvas.cd(2); hChannelHits_PSide->Draw();
    }, 1200, 800);
    
    create_and_save_plot(plotsDir, "nside_pside_hits_log", [&]() {
        TCanvas canvas("nside_pside_hits_log", "N-side vs P-side Hits Log", 1200, 800);
        canvas.Divide(2, 1);
        canvas.cd(1)->SetLogy(); hChannelHits_NSideLog->Draw();
        canvas.cd(2)->SetLogy(); hChannelHits_PSideLog->Draw();
    }, 1200, 800);
    
    // Mean flux distribution plots
    create_and_save_plot(plotsDir, "mean_flux_distribution", [&]() {
        TCanvas canvas("mean_flux_distribution", "Mean Flux Distribution", 1200, 800);
        canvas.Divide(2, 2);
        canvas.cd(1); hMeanFluxDistribution->Draw();
        canvas.cd(2); hMeanFluxDistributionNSide->Draw();
        canvas.cd(3); hMeanFluxDistributionPSide->Draw();
        // Leave 4th pad empty for better layout
    }, 1200, 800);
    
    // N-side vs P-side mean flux comparison
    create_and_save_plot(plotsDir, "nside_pside_mean_flux", [&]() {
        TCanvas canvas("nside_pside_mean_flux", "N-side vs P-side Mean Flux", 1200, 800);
        canvas.Divide(2, 1);
        canvas.cd(1); 
        hMeanChargeFluxNSide->SetLineColor(kBlue);
        hMeanChargeFluxNSide->Draw();
        
        canvas.cd(2);
        hMeanChargeFluxPSide->SetLineColor(kRed);
        hMeanChargeFluxPSide->Draw();
    }, 1200, 800);
    
    // Time interval plots
    create_and_save_plot(plotsDir, "time_intervals", [&]() {
        TCanvas canvas("time_intervals", "Time Intervals", 1200, 800);
        canvas.Divide(2, 2);
        
        canvas.cd(1); 
        hTimeIntervalGlobal->SetLineColor(kBlack);
        hTimeIntervalGlobal->Draw();
        
        canvas.cd(2);
        hTimeIntervalNSide->SetLineColor(kBlue);
        hTimeIntervalNSide->Draw();
        
        canvas.cd(3);
        hTimeIntervalPSide->SetLineColor(kRed);
        hTimeIntervalPSide->Draw();
        
        canvas.cd(4);
        gPad->SetLogy();
        hTimeIntervalLog->SetLineColor(kGreen+2);
        hTimeIntervalLog->Draw();
    }, 1200, 800);
    
    // Time interval comparison plot
    create_and_save_plot(plotsDir, "time_interval_comparison", [&]() {
        TCanvas canvas("time_interval_comparison", "Time Interval Comparison", 800, 600);
        
        // Set styles
        hTimeIntervalGlobal->SetLineColor(kBlack);
        hTimeIntervalGlobal->SetLineWidth(2);
        hTimeIntervalNSide->SetLineColor(kBlue);
        hTimeIntervalNSide->SetLineWidth(2);
        hTimeIntervalPSide->SetLineColor(kRed);
        hTimeIntervalPSide->SetLineWidth(2);
        
        // Draw with axis range adjustment
        double maxY = std::max({hTimeIntervalGlobal->GetMaximum(), 
                               hTimeIntervalNSide->GetMaximum(), 
                               hTimeIntervalPSide->GetMaximum()});
        
        hTimeIntervalGlobal->GetYaxis()->SetRangeUser(0, maxY * 1.1);
        hTimeIntervalGlobal->Draw();
        hTimeIntervalNSide->Draw("SAME");
        hTimeIntervalPSide->Draw("SAME");
        
        // Add legend
        TLegend leg(0.7, 0.7, 0.9, 0.9);
        leg.SetBorderSize(0);
        leg.SetFillStyle(0);
        leg.AddEntry(hTimeIntervalGlobal, "Global", "l");
        leg.AddEntry(hTimeIntervalNSide, "N-side", "l");
        leg.AddEntry(hTimeIntervalPSide, "P-side", "l");
        leg.Draw();
    });
}

// Function to create comprehensive comparison plots
void create_comparison_plots(const TString& plotsDir,
                           const std::vector<double>& nsideFlux, const std::vector<double>& psideFlux,
                           const std::vector<double>& vreftValues, const std::vector<double>& vrefzValues) {
    
    if (nsideFlux.empty() || psideFlux.empty()) return;
    
    // Create ratio plots
    std::vector<double> ratios;
    std::vector<double> ratioVreft;
    std::vector<double> ratioVrefz;
    
    for (size_t i = 0; i < nsideFlux.size(); i++) {
        if (psideFlux[i] > 0) {
            double ratio = nsideFlux[i] / psideFlux[i];
            ratios.push_back(ratio);
            ratioVreft.push_back(vreftValues[i]);
            ratioVrefz.push_back(vrefzValues[i]);
        }
    }
    
    if (ratios.empty()) return;
    
    // Ratio vs Vreft
    TGraph ratioVsVreft(ratios.size(), ratioVreft.data(), ratios.data());
    ratioVsVreft.SetTitle("N-side/P-side Ratio vs Vreft;Vreft;Ratio (N-side/P-side)");
    ratioVsVreft.SetMarkerStyle(20);
    ratioVsVreft.SetMarkerColor(kBlue);
    
    create_and_save_plot(plotsDir, "ratio_vs_vreft", [&]() {
        ratioVsVreft.Draw("AP");
        gPad->SetGrid(true, true);
    });
    
    // Ratio vs Vrefz
    TGraph ratioVsVrefz(ratios.size(), ratioVrefz.data(), ratios.data());
    ratioVsVrefz.SetTitle("N-side/P-side Ratio vs VrefT_z-strips;VrefT_z-strips;Ratio (N-side/P-side)");
    ratioVsVrefz.SetMarkerStyle(20);
    ratioVsVrefz.SetMarkerColor(kRed);
    
    create_and_save_plot(plotsDir, "ratio_vs_vrefz", [&]() {
        ratioVsVrefz.Draw("AP");
        gPad->SetGrid(true, true);
    });
    
    // Comprehensive comparison plot
    create_and_save_plot(plotsDir, "comprehensive_comparison", [&]() {
        TCanvas canvas("comprehensive_comparison", "Comprehensive Comparison", 1600, 1200);
        canvas.Divide(2, 3);
        
        // N-side flux vs Vreft
        canvas.cd(1);
        TGraph nsideVsVreft(nsideFlux.size(), vreftValues.data(), nsideFlux.data());
        nsideVsVreft.SetTitle("N-side Flux vs Vreft;Vreft;N-side Flux");
        nsideVsVreft.SetMarkerStyle(20);
        nsideVsVreft.SetMarkerColor(kBlue);
        nsideVsVreft.Draw("AP");
        gPad->SetGrid(true, true);
        
        // P-side flux vs Vreft
        canvas.cd(2);
        TGraph psideVsVreft(psideFlux.size(), vreftValues.data(), psideFlux.data());
        psideVsVreft.SetTitle("P-side Flux vs Vreft;Vreft;P-side Flux");
        psideVsVreft.SetMarkerStyle(21);
        psideVsVreft.SetMarkerColor(kRed);
        psideVsVreft.Draw("AP");
        gPad->SetGrid(true, true);
        
        // N-side flux vs Vrefz
        canvas.cd(3);
        TGraph nsideVsVrefz(nsideFlux.size(), vrefzValues.data(), nsideFlux.data());
        nsideVsVrefz.SetTitle("N-side Flux vs VrefT_z;VrefT_z;N-side Flux");
        nsideVsVrefz.SetMarkerStyle(20);
        nsideVsVrefz.SetMarkerColor(kBlue);
        nsideVsVrefz.Draw("AP");
        gPad->SetGrid(true, true);
        
        // P-side flux vs Vrefz
        canvas.cd(4);
        TGraph psideVsVrefz(psideFlux.size(), vrefzValues.data(), psideFlux.data());
        psideVsVrefz.SetTitle("P-side Flux vs VrefT_z;VrefT_z;P-side Flux");
        psideVsVrefz.SetMarkerStyle(21);
        psideVsVrefz.SetMarkerColor(kRed);
        psideVsVrefz.Draw("AP");
        gPad->SetGrid(true, true);
        
        // Ratio plots
        canvas.cd(5);
        ratioVsVreft.Draw("AP");
        gPad->SetGrid(true, true);
        
        canvas.cd(6);
        ratioVsVrefz.Draw("AP");
        gPad->SetGrid(true, true);
        
    }, 1600, 1200);
}

// Function to create 3D visualization plots
void create_3d_visualization_plots(const TString& plotsDir,
                                  const std::vector<double>& xValues, const std::vector<double>& yValues, 
                                  const std::vector<double>& zValues, const TString& title) {
    
    if (xValues.empty()) return;
    
    // Create 3D scatter plot
    create_and_save_plot(plotsDir, "3d_scatter_" + title, [&]() {
        TCanvas canvas("3d_scatter", "3D Scatter Plot", 1000, 800);
        
        // Convert vectors to arrays
        std::vector<Double_t> xD(xValues.begin(), xValues.end());
        std::vector<Double_t> yD(yValues.begin(), yValues.end());
        std::vector<Double_t> zD(zValues.begin(), zValues.end());
        
        TGraph2D graph3d(xValues.size(), xD.data(), yD.data(), zD.data());
        graph3d.SetTitle(Form("%s;Vreft;VrefT_z;Flux", title.Data()));
        graph3d.Draw("pcol");
        gPad->SetTheta(30);
        gPad->SetPhi(30);
    }, 1000, 800);
    
    // Create 2D projection plots
    create_and_save_plot(plotsDir, "2d_projection_" + title, [&]() {
        TCanvas canvas("2d_projection", "2D Projection", 1200, 800);
        canvas.Divide(2, 2);
        
        // X vs Y
        canvas.cd(1);
        TH2F hxy("hxy", "X vs Y Projection;Vreft;VrefT_z", 
                50, *std::min_element(xValues.begin(), xValues.end()), 
                *std::max_element(xValues.begin(), xValues.end()),
                50, *std::min_element(yValues.begin(), yValues.end()), 
                *std::max_element(yValues.begin(), yValues.end()));
        for (size_t i = 0; i < xValues.size(); i++) {
            hxy.Fill(xValues[i], yValues[i]);
        }
        hxy.Draw("colz");
        
        // X vs Z
        canvas.cd(2);
        std::vector<Double_t> xD(xValues.begin(), xValues.end());
        std::vector<Double_t> zD(zValues.begin(), zValues.end());
        TGraph xzGraph(xValues.size(), xD.data(), zD.data());
        xzGraph.SetTitle("X vs Z;Vreft;Flux");
        xzGraph.SetMarkerStyle(20);
        xzGraph.Draw("AP");
        gPad->SetGrid(true, true);
        
        // Y vs Z
        canvas.cd(3);
        std::vector<Double_t> yD(yValues.begin(), yValues.end());
        TGraph yzGraph(yValues.size(), yD.data(), zD.data());
        yzGraph.SetTitle("Y vs Z;VrefT_z;Flux");
        yzGraph.SetMarkerStyle(20);
        yzGraph.Draw("AP");
        gPad->SetGrid(true, true);
        
        // Z distribution
        canvas.cd(4);
        TH1F hz("hz", "Flux Distribution;Flux;Count", 50, 
                *std::min_element(zValues.begin(), zValues.end()), 
                *std::max_element(zValues.begin(), zValues.end()));
        for (double z : zValues) hz.Fill(z);
        hz.Draw();
        
    }, 1200, 800);
}

// Function to create calibration correlation plots
void create_calibration_plots(const TString& plotsDir,
                             const std::vector<AnalysisResult>& results) {
    
    std::vector<double> vreftValues, vrefzValues, fluxValues, nsideFlux, psideFlux;
    std::vector<Double_t> runNumbers;
    
    for (const auto& result : results) {
        if (result.Vreft != -999 && result.VrefT_z_strips != -999) {
            vreftValues.push_back(result.Vreft);
            vrefzValues.push_back(result.VrefT_z_strips);
            fluxValues.push_back(result.mean_charge_flux);
            nsideFlux.push_back(result.mean_charge_flux_nside);
            psideFlux.push_back(result.mean_charge_flux_pside);
            runNumbers.push_back(result.runNumber);
        }
    }
    
    if (vreftValues.empty()) return;
    
    // Create flux vs run number plot
    create_and_save_plot(plotsDir, "flux_vs_run", [&]() {
        std::vector<Double_t> runNumbersD(runNumbers.begin(), runNumbers.end());
        std::vector<Double_t> fluxValuesD(fluxValues.begin(), fluxValues.end());
        
        TGraph fluxVsRun(runNumbers.size(), runNumbersD.data(), fluxValuesD.data());
        fluxVsRun.SetTitle("Flux vs Run Number;Run Number;Mean Charge Flux");
        fluxVsRun.SetMarkerStyle(20);
        fluxVsRun.Draw("AP");
        gPad->SetGrid(true, true);
    });
    
    // Create calibration parameter plots
    create_and_save_plot(plotsDir, "calibration_correlations", [&]() {
        TCanvas canvas("calibration_correlations", "Calibration Correlations", 1200, 800);
        canvas.Divide(2, 2);
        
        // Flux vs Vreft
        canvas.cd(1);
        std::vector<Double_t> vreftD(vreftValues.begin(), vreftValues.end());
        std::vector<Double_t> fluxD(fluxValues.begin(), fluxValues.end());
        TGraph fluxVsVreft(vreftValues.size(), vreftD.data(), fluxD.data());
        fluxVsVreft.SetTitle("Flux vs Vreft;Vreft;Flux");
        fluxVsVreft.SetMarkerStyle(20);
        fluxVsVreft.Draw("AP");
        gPad->SetGrid(true, true);
        
        // Flux vs Vrefz
        canvas.cd(2);
        std::vector<Double_t> vrefzD(vrefzValues.begin(), vrefzValues.end());
        TGraph fluxVsVrefz(vrefzValues.size(), vrefzD.data(), fluxD.data());
        fluxVsVrefz.SetTitle("Flux vs VrefT_z;VrefT_z;Flux");
        fluxVsVrefz.SetMarkerStyle(20);
        fluxVsVrefz.Draw("AP");
        gPad->SetGrid(true, true);
        
        // N-side/P-side comparison
        canvas.cd(3);
        TH2F frame("frame", "N-side vs P-side Flux;N-side Flux;P-side Flux", 
                  100, 0, *std::max_element(nsideFlux.begin(), nsideFlux.end()) * 1.1,
                  100, 0, *std::max_element(psideFlux.begin(), psideFlux.end()) * 1.1);
        frame.Draw();
        
        std::vector<Double_t> nsideD(nsideFlux.begin(), nsideFlux.end());
        std::vector<Double_t> psideD(psideFlux.begin(), psideFlux.end());
        TGraph nsideVsPside(nsideFlux.size(), nsideD.data(), psideD.data());
        nsideVsPside.SetMarkerStyle(20);
        nsideVsPside.Draw("P SAME");
        
        // Add unity line for reference
        TLine unity(0, 0, frame.GetXaxis()->GetXmax(), frame.GetXaxis()->GetXmax());
        unity.SetLineColor(kRed);
        unity.SetLineStyle(2);
        unity.Draw();
        
        canvas.cd(4);
        std::vector<double> ratios;
        for (size_t i = 0; i < nsideFlux.size(); i++) {
            if (psideFlux[i] > 0) {
                ratios.push_back(nsideFlux[i] / psideFlux[i]);
            }
        }
        TH1F ratioHist("ratioHist", "N-side/P-side Ratio Distribution;Ratio;Count", 
                      50, 0, *std::max_element(ratios.begin(), ratios.end()) * 1.1);
        for (double ratio : ratios) ratioHist.Fill(ratio);
        ratioHist.Draw();
        
    }, 1200, 800);
}

// Function to analyze a single file
void analyze_single_file(const TString& inputFile, const TString& outputBaseDir, 
                        const std::map<int, CalibrationData>& calibrationMap) {
    int runNumber = extract_run_number(inputFile);
    gLastHitTime[runNumber].clear();
    
    // Get calibration data
    CalibrationData calibData;
    auto it = calibrationMap.find(runNumber);
    if (it != calibrationMap.end()) {
        calibData = it->second;
        std::cout << "Found calibration data for run " << runNumber << std::endl;
    } else {
        std::cout << "Warning: No calibration data found for run " << runNumber << std::endl;
        calibData.runNumber = runNumber;
    }
    
    TString baseName = gSystem->BaseName(inputFile);
    baseName.ReplaceAll(".digi.root", "");
    TString outputDir = outputBaseDir + "/" + baseName;
    TString plotsDir = outputDir + "/plots";
    gSystem->mkdir(outputDir, kTRUE);
    gSystem->mkdir(plotsDir, kTRUE);
    
    std::cout << "Analyzing: " << inputFile << ", Run: " << runNumber << std::endl;
    std::cout << "Output directory: " << outputDir << std::endl;
    std::cout << "Plots directory: " << plotsDir << std::endl;
    
    // Open input file
    std::unique_ptr<TFile> file(TFile::Open(inputFile));
    if (!file || file->IsZombie()) {
        std::cerr << "Error: Could not open file: " << inputFile << std::endl;
        return;
    }
    
    TTree* tree = dynamic_cast<TTree*>(file->Get("cbmsim"));
    if (!tree) {
        std::cerr << "Error: Could not find tree 'cbmsim'" << std::endl;
        return;
    }
    
    std::vector<CbmStsDigi>* digiVector = nullptr;
    tree->SetBranchAddress("StsDigi", &digiVector);
    
    Long64_t nEntries = tree->GetEntries();
    std::cout << "Events: " << nEntries << std::endl;
    
    if (nEntries == 0) {
        return;
    }
    
    // Pre-allocate arrays
    std::vector<double> totalChargePerChannel(2048, 0.0);
    std::vector<int> hitsPerChannel(2048, 0);
    std::vector<double> totalChargeNSide(1024, 0.0);
    std::vector<double> totalChargePSide(1024, 0.0);
    std::vector<int> hitsNSide(1024, 0);
    std::vector<int> hitsPSide(1024, 0);
    
    int maxChargeValue = 0;
    
    // First pass: find maximum charge and collect basic statistics
    for (Long64_t iEntry = 0; iEntry < nEntries; iEntry++) {
        tree->GetEntry(iEntry);
        
        if (iEntry % 10 == 0) {
            std::cout << "Processing event " << iEntry << "/" << nEntries << std::endl;
        }
        
        for (const auto& digi : *digiVector) {
            int charge = digi.GetCharge();
            if (charge > maxChargeValue) maxChargeValue = charge;
        }
    }
    
    // Create histograms with smart pointers for automatic cleanup
    int nBinsCharge = std::min(maxChargeValue + 100, 10000); // Limit maximum bins
    
    std::unique_ptr<TH1F> hTotalCharge(new TH1F("hTotalCharge", "Charge distribution;Charge;Count", 
                                               nBinsCharge, 0, nBinsCharge));
    std::unique_ptr<TH1F> hMeanChargeFlux(new TH1F("hMeanChargeFlux", "Mean charge flux per channel;Channel;Mean charge flux", 
                                                  2048, 0, 2048));
    std::unique_ptr<TH1F> hChannelHits(new TH1F("hChannelHits", "Number of hits per channel;Channel;Number of hits", 
                                               2048, 0, 2048));
    std::unique_ptr<TH1F> hChannelHits_NSide(new TH1F("hChannelHits_NSide", "Number of hits per channel (N-side);Channel;Number of hits", 
                                                     1024, 0, 1024));
    std::unique_ptr<TH1F> hChannelHits_PSide(new TH1F("hChannelHits_PSide", "Number of hits per channel (P-side);Channel;Number of hits", 
                                                     1024, 0, 1024));
    
    // Second pass: process data
    for (Long64_t iEntry = 0; iEntry < nEntries; iEntry++) {
        tree->GetEntry(iEntry);
        
        for (const auto& digi : *digiVector) {
            int channel = digi.GetChannel();
            int charge = digi.GetCharge();
            
            // Calculate time intervals
            calculate_time_intervals(runNumber, *digiVector);
            
            // Ensure charge is non-negative
            if (charge < 0) charge = 0;
            
            if (channel >= 0 && channel < 2048) {
                hTotalCharge->Fill(charge);
                totalChargePerChannel[channel] += charge;
                hitsPerChannel[channel]++;
                
                // Fill channel hits histograms
                hChannelHits->Fill(channel);
                
                // Separate N-side and P-side accumulation
                if (channel < 1024) {
                    hChannelHits_NSide->Fill(channel);
                    totalChargeNSide[channel] += charge;
                    hitsNSide[channel]++;
                } else {
                    int psideChannel = channel - 1024;
                    hChannelHits_PSide->Fill(psideChannel);
                    totalChargePSide[psideChannel] += charge;
                    hitsPSide[psideChannel]++;
                }
            }
        }
    }
    
    // Calculate mean flux values
    double overallMeanFlux = 0.0;
    double overallMeanFluxNSide = 0.0;
    double overallMeanFluxPSide = 0.0;
    int channelsWithHits = 0;
    int channelsWithHitsNSide = 0;
    int channelsWithHitsPSide = 0;
    
    std::vector<double> meanChargeFluxPerChannel(2048, 0.0);
    std::vector<double> meanFluxNSide(1024, 0.0);
    std::vector<double> meanFluxPSide(1024, 0.0);
    
    for (int channel = 0; channel < 2048; channel++) {
        if (hitsPerChannel[channel] > 0 && totalChargePerChannel[channel] >= 0) {
            meanChargeFluxPerChannel[channel] = totalChargePerChannel[channel] / hitsPerChannel[channel];
            overallMeanFlux += meanChargeFluxPerChannel[channel];
            channelsWithHits++;
            hMeanChargeFlux->SetBinContent(channel + 1, meanChargeFluxPerChannel[channel]);
        }
    }
    
    for (int channel = 0; channel < 1024; channel++) {
        if (hitsNSide[channel] > 0 && totalChargeNSide[channel] >= 0) {
            meanFluxNSide[channel] = totalChargeNSide[channel] / hitsNSide[channel];
            overallMeanFluxNSide += meanFluxNSide[channel];
            channelsWithHitsNSide++;
        }
        if (hitsPSide[channel] > 0 && totalChargePSide[channel] >= 0) {
            meanFluxPSide[channel] = totalChargePSide[channel] / hitsPSide[channel];
            overallMeanFluxPSide += meanFluxPSide[channel];
            channelsWithHitsPSide++;
        }
    }
    
    // Calculate averages
    if (channelsWithHits > 0) overallMeanFlux /= channelsWithHits;
    if (channelsWithHitsNSide > 0) overallMeanFluxNSide /= channelsWithHitsNSide;
    if (channelsWithHitsPSide > 0) overallMeanFluxPSide /= channelsWithHitsPSide;
    
    // Calculate active channels
    int activeChannels = 0;
    int activeChannels_NSide = 0;
    int activeChannels_PSide = 0;
    for (int i = 0; i < 2048; i++) {
        if (hitsPerChannel[i] > 0) {
            activeChannels++;
            if (i < 1024) activeChannels_NSide++;
            else activeChannels_PSide++;
        }
    }
    
    // Create time interval histograms
    const double maxInterval = 1000.0;
    std::unique_ptr<TH1F> hTimeIntervalGlobal(new TH1F("hTimeIntervalGlobal", 
        "Time intervals between signals (Global);Time Interval [ns];Count", 200, 0, maxInterval));
    std::unique_ptr<TH1F> hTimeIntervalNSide(new TH1F("hTimeIntervalNSide", 
        "Time intervals between signals (N-side);Time Interval [ns];Count", 200, 0, maxInterval));
    std::unique_ptr<TH1F> hTimeIntervalPSide(new TH1F("hTimeIntervalPSide", 
        "Time intervals between signals (P-side);Time Interval [ns];Count", 200, 0, maxInterval));
    std::unique_ptr<TH1F> hTimeIntervalLog(new TH1F("hTimeIntervalLog", 
        "Time intervals between signals (Log scale);Time Interval [ns];Count", 200, 0, maxInterval));
    
    // Fill time interval histograms
    for (double interval : gTimeIntervals.global_intervals) {
        hTimeIntervalGlobal->Fill(interval);
        hTimeIntervalLog->Fill(interval);
    }
    for (double interval : gTimeIntervals.nside_intervals) {
        hTimeIntervalNSide->Fill(interval);
    }
    for (double interval : gTimeIntervals.pside_intervals) {
        hTimeIntervalPSide->Fill(interval);
    }
    
    // Calculate time interval statistics
    double meanIntervalGlobal = hTimeIntervalGlobal->GetMean();
    double rmsIntervalGlobal = hTimeIntervalGlobal->GetRMS();
    double meanIntervalNSide = hTimeIntervalNSide->GetMean();
    double meanIntervalPSide = hTimeIntervalPSide->GetMean();
    
    // Store results
    AnalysisResult result;
    result.runNumber = runNumber;
    result.mean_charge = hTotalCharge->GetMean();
    result.rms_charge = hTotalCharge->GetRMS();
    result.mean_charge_flux = overallMeanFlux;
    result.mean_charge_flux_nside = overallMeanFluxNSide;
    result.mean_charge_flux_pside = overallMeanFluxPSide;
    result.total_hits = hTotalCharge->GetEntries();
    result.active_channels = activeChannels;
    result.active_channels_nside = activeChannels_NSide;
    result.active_channels_pside = activeChannels_PSide;
    result.Vreft = calibData.Vreft;
    result.VrefT_z_strips = calibData.VrefT_z_strips;
    result.cas_csa = calibData.cas_csa;
    result.calibration_params = calibData.calibration_params;
    result.cas_shap = calibData.cas_shap;
    result.comments = calibData.comments;
    
    gAnalysisResults.push_back(result);
    
    // Create log plots
    const double SMALL_VALUE = 0.1;
    std::unique_ptr<TH1F> hChannelHitsLog(new TH1F("hChannelHitsLog", "Number of hits per channel (log scale);Channel;Number of hits", 2048, 0, 2048));
    std::unique_ptr<TH1F> hChannelHits_NSideLog(new TH1F("hChannelHits_NSideLog", "Number of hits per channel (N-side, log scale);Channel;Number of hits", 1024, 0, 1024));
    std::unique_ptr<TH1F> hChannelHits_PSideLog(new TH1F("hChannelHits_PSideLog", "Number of hits per channel (P-side, log scale);Channel;Number of hits", 1024, 0, 1024));
    
    for (int channel = 0; channel < 2048; channel++) {
        double value = hitsPerChannel[channel] > 0 ? hitsPerChannel[channel] : SMALL_VALUE;
        hChannelHitsLog->SetBinContent(channel + 1, value);
        
        if (channel < 1024) {
            hChannelHits_NSideLog->SetBinContent(channel + 1, value);
        } else {
            hChannelHits_PSideLog->SetBinContent(channel - 1024 + 1, value);
        }
    }
    
    // Create mean flux distribution histograms
    double fluxMaxOverall = 0.0;
    double fluxMaxNSide = 0.0;
    double fluxMaxPSide = 0.0;
    
    for (int channel = 0; channel < 2048; channel++) {
        if (meanChargeFluxPerChannel[channel] > fluxMaxOverall) 
            fluxMaxOverall = meanChargeFluxPerChannel[channel];
    }
    for (int channel = 0; channel < 1024; channel++) {
        if (meanFluxNSide[channel] > fluxMaxNSide) 
            fluxMaxNSide = meanFluxNSide[channel];
        if (meanFluxPSide[channel] > fluxMaxPSide) 
            fluxMaxPSide = meanFluxPSide[channel];
    }
    
    fluxMaxOverall = std::max(fluxMaxOverall * 1.1, 10.0);
    fluxMaxNSide = std::max(fluxMaxNSide * 1.1, 10.0);
    fluxMaxPSide = std::max(fluxMaxPSide * 1.1, 10.0);
    
    std::unique_ptr<TH1F> hMeanFluxDistribution(new TH1F("hMeanFluxDistribution", 
        "Distribution of Mean Charge Flux per Channel;Mean Charge Flux;Number of Channels",
        100, 0, fluxMaxOverall));
    std::unique_ptr<TH1F> hMeanFluxDistributionNSide(new TH1F("hMeanFluxDistributionNSide", 
        "Distribution of Mean Charge Flux per Channel (N-side);Mean Charge Flux;Number of Channels",
        100, 0, fluxMaxNSide));
    std::unique_ptr<TH1F> hMeanFluxDistributionPSide(new TH1F("hMeanFluxDistributionPSide", 
        "Distribution of Mean Charge Flux per Channel (P-side);Mean Charge Flux;Number of Channels",
        100, 0, fluxMaxPSide));
    
    for (int channel = 0; channel < 2048; channel++) {
        if (hitsPerChannel[channel] > 0) {
            hMeanFluxDistribution->Fill(meanChargeFluxPerChannel[channel]);
        }
    }
    for (int channel = 0; channel < 1024; channel++) {
        if (hitsNSide[channel] > 0) {
            hMeanFluxDistributionNSide->Fill(meanFluxNSide[channel]);
        }
        if (hitsPSide[channel] > 0) {
            hMeanFluxDistributionPSide->Fill(meanFluxPSide[channel]);
        }
    }
    
    // Create mean flux per channel histograms
    std::unique_ptr<TH1F> hMeanChargeFluxNSide(new TH1F("hMeanChargeFluxNSide", 
        "Mean charge flux per channel (N-side);Channel;Mean charge flux", 1024, 0, 1024));
    std::unique_ptr<TH1F> hMeanChargeFluxPSide(new TH1F("hMeanChargeFluxPSide", 
        "Mean charge flux per channel (P-side);Channel;Mean charge flux", 1024, 0, 1024));
    
    for (int channel = 0; channel < 1024; channel++) {
        hMeanChargeFluxNSide->SetBinContent(channel + 1, meanFluxNSide[channel]);
        hMeanChargeFluxPSide->SetBinContent(channel + 1, meanFluxPSide[channel]);
    }
    
    // Create and save plots
    std::vector<std::unique_ptr<TCanvas>> canvases;
    
    auto create_and_save_plot = [&](const TString& name, auto drawFunc) {
        canvases.emplace_back(std::make_unique<TCanvas>(name, name, 800, 600));
        drawFunc();
        canvases.back()->SaveAs(Form("%s/%s.png", plotsDir.Data(), name.Data()));
    };
    
    // Individual plots
    create_and_save_plot("charge_distribution", [&]() { hTotalCharge->Draw(); });
    create_and_save_plot("mean_charge_flux", [&]() { hMeanChargeFlux->Draw(); });
    create_and_save_plot("channel_hits", [&]() { hChannelHits->Draw(); });
    
    create_and_save_plot("channel_hits_log", [&]() { 
        canvases.back()->SetLogy();
        hChannelHitsLog->Draw(); 
    });
    
    // N-side vs P-side plots
    canvases.emplace_back(std::make_unique<TCanvas>("nside_pside_hits", "N-side vs P-side Hits", 1200, 800));
    canvases.back()->Divide(2, 1);
    canvases.back()->cd(1); hChannelHits_NSide->Draw();
    canvases.back()->cd(2); hChannelHits_PSide->Draw();
    canvases.back()->SaveAs(Form("%s/nside_pside_hits.png", plotsDir.Data()));
    
    canvases.emplace_back(std::make_unique<TCanvas>("nside_pside_hits_log", "N-side vs P-side Hits Log", 1200, 800));
    canvases.back()->Divide(2, 1);
    canvases.back()->cd(1)->SetLogy(); hChannelHits_NSideLog->Draw();
    canvases.back()->cd(2)->SetLogy(); hChannelHits_PSideLog->Draw();
    canvases.back()->SaveAs(Form("%s/nside_pside_hits_log.png", plotsDir.Data()));
    
    // Mean flux distribution plots
    canvases.emplace_back(std::make_unique<TCanvas>("mean_flux_distribution", "Mean Flux Distribution", 1200, 800));
    canvases.back()->Divide(2, 2);
    canvases.back()->cd(1); hMeanFluxDistribution->Draw();
    canvases.back()->cd(2); hMeanFluxDistributionNSide->Draw();
    canvases.back()->cd(3); hMeanFluxDistributionPSide->Draw();
    canvases.back()->SaveAs(Form("%s/mean_flux_distribution.png", plotsDir.Data()));
    
    // N-side vs P-side mean flux
    canvases.emplace_back(std::make_unique<TCanvas>("nside_pside_mean_flux", "N-side vs P-side Mean Flux", 1200, 800));
    canvases.back()->Divide(2, 1);
    canvases.back()->cd(1); hMeanChargeFluxNSide->Draw();
    canvases.back()->cd(2); hMeanChargeFluxPSide->Draw();
    canvases.back()->SaveAs(Form("%s/nside_pside_mean_flux.png", plotsDir.Data()));
    
    // Time interval plots
    canvases.emplace_back(std::make_unique<TCanvas>("time_intervals", "Time Intervals", 1200, 800));
    canvases.back()->Divide(2, 2);
    canvases.back()->cd(1); hTimeIntervalGlobal->Draw();
    canvases.back()->cd(2); hTimeIntervalNSide->Draw(); hTimeIntervalNSide->SetLineColor(kBlue);
    canvases.back()->cd(3); hTimeIntervalPSide->Draw(); hTimeIntervalPSide->SetLineColor(kRed);
    canvases.back()->cd(4)->SetLogy(); hTimeIntervalLog->Draw();
    canvases.back()->SaveAs(Form("%s/time_intervals.png", plotsDir.Data()));
    
    // Time interval comparison
    canvases.emplace_back(std::make_unique<TCanvas>("time_interval_comparison", "Time Interval Comparison", 800, 600));
    hTimeIntervalGlobal->Draw();
    hTimeIntervalNSide->Draw("SAME");
    hTimeIntervalPSide->Draw("SAME");
    
    std::unique_ptr<TLegend> legTime(new TLegend(0.7, 0.7, 0.9, 0.9));
    legTime->AddEntry(hTimeIntervalGlobal.get(), "Global", "l");
    legTime->AddEntry(hTimeIntervalNSide.get(), "N-side", "l");
    legTime->AddEntry(hTimeIntervalPSide.get(), "P-side", "l");
    legTime->Draw();
    canvases.back()->SaveAs(Form("%s/time_interval_comparison.png", plotsDir.Data()));
    
    // Save results to file
    std::unique_ptr<TFile> outputFile(new TFile(Form("%s/charge_analysis_%s.root", outputDir.Data(), baseName.Data()), "RECREATE"));
    
    if (outputFile && !outputFile->IsZombie()) {
        TTree resultsTree("analysis_results", "Analysis Results with Calibration Data");
        
        Int_t run_number = runNumber;
        Double_t mean_charge = hTotalCharge->GetMean();
        Double_t rms_charge = hTotalCharge->GetRMS();
        Double_t mean_charge_flux = overallMeanFlux;
        Double_t mean_charge_flux_nside = overallMeanFluxNSide;
        Double_t mean_charge_flux_pside = overallMeanFluxPSide;
        Int_t total_hits = hTotalCharge->GetEntries();
        Int_t active_channels = activeChannels;
        Int_t active_channels_nside = activeChannels_NSide;
        Int_t active_channels_pside = activeChannels_PSide;
        Int_t calib_Vreft = calibData.Vreft;
        Int_t calib_VrefT_z_strips = calibData.VrefT_z_strips;
        Int_t calib_cas_csa = calibData.cas_csa;
        TString calib_params = calibData.calibration_params;
        Int_t calib_cas_shap = calibData.cas_shap;
        TString calib_comments = calibData.comments;
        
        resultsTree.Branch("run_number", &run_number);
        resultsTree.Branch("mean_charge", &mean_charge);
        resultsTree.Branch("rms_charge", &rms_charge);
        resultsTree.Branch("mean_charge_flux", &mean_charge_flux);
        resultsTree.Branch("mean_charge_flux_nside", &mean_charge_flux_nside);
        resultsTree.Branch("mean_charge_flux_pside", &mean_charge_flux_pside);
        resultsTree.Branch("total_hits", &total_hits);
        resultsTree.Branch("active_channels", &active_channels);
        resultsTree.Branch("active_channels_nside", &active_channels_nside);
        resultsTree.Branch("active_channels_pside", &active_channels_pside);
        resultsTree.Branch("calib_Vreft", &calib_Vreft);
        resultsTree.Branch("calib_VrefT_z_strips", &calib_VrefT_z_strips);
        resultsTree.Branch("calib_cas_csa", &calib_cas_csa);
        resultsTree.Branch("calib_params", &calib_params);
        resultsTree.Branch("calib_cas_shap", &calib_cas_shap);
        resultsTree.Branch("calib_comments", &calib_comments);
        
        resultsTree.Fill();
        
        // Write all histograms
        hTotalCharge->Write();
        hMeanChargeFlux->Write();
        hChannelHits->Write();
        hChannelHits_NSide->Write();
        hChannelHits_PSide->Write();
        hChannelHitsLog->Write();
        hChannelHits_NSideLog->Write();
        hChannelHits_PSideLog->Write();
        hMeanFluxDistribution->Write();
        hMeanFluxDistributionNSide->Write();
        hMeanFluxDistributionPSide->Write();
        hMeanChargeFluxNSide->Write();
        hMeanChargeFluxPSide->Write();
        resultsTree.Write();
        hTimeIntervalGlobal->Write();
        hTimeIntervalNSide->Write();
        hTimeIntervalPSide->Write();
        hTimeIntervalLog->Write();
        
        outputFile->Close();
    }
    
    // Create summary file
    std::ofstream summaryFile(Form("%s/summary_%s.txt", outputDir.Data(), baseName.Data()));
    if (summaryFile.is_open()) {
        summaryFile << "=== CHARGE ANALYSIS SUMMARY ===" << std::endl;
        summaryFile << "Input file: " << inputFile.Data() << std::endl;
        summaryFile << "Run number: " << runNumber << std::endl;
        summaryFile << "=== CALIBRATION DATA ===" << std::endl;
        summaryFile << "Vreft: " << calibData.Vreft << std::endl;
        summaryFile << "VrefT_z-strips: " << (calibData.VrefT_z_strips != -999 ? std::to_string(calibData.VrefT_z_strips) : "N/A") << std::endl;
        summaryFile << "cas_csa: " << calibData.cas_csa << std::endl;
        summaryFile << "Calibration parameters: " << calibData.calibration_params.Data() << std::endl;
        summaryFile << "cas_shap: " << (calibData.cas_shap != -999 ? std::to_string(calibData.cas_shap) : "N/A") << std::endl;
        summaryFile << "Comments: " << calibData.comments.Data() << std::endl;
        summaryFile << "=== ANALYSIS RESULTS ===" << std::endl;
        summaryFile << "Number of events: " << nEntries << std::endl;
        summaryFile << "Total hits: " << hTotalCharge->GetEntries() << std::endl;
        summaryFile << "Mean charge: " << hTotalCharge->GetMean() << " ± " << hTotalCharge->GetRMS() << std::endl;
        summaryFile << "Mean charge flux: " << overallMeanFlux << std::endl;
        summaryFile << "Mean charge flux (N-side): " << overallMeanFluxNSide << std::endl;
        summaryFile << "Mean charge flux (P-side): " << overallMeanFluxPSide << std::endl;
        summaryFile << "N-side/P-side ratio: " << (overallMeanFluxPSide > 0 ? overallMeanFluxNSide/overallMeanFluxPSide : 0) << std::endl;
        summaryFile << "Active channels: " << activeChannels << "/2048 (" << (100.0 * activeChannels / 2048) << "%)" << std::endl;
        summaryFile << "Active N-side channels: " << activeChannels_NSide << "/1024 (" << (100.0 * activeChannels_NSide / 1024) << "%)" << std::endl;
        summaryFile << "Active P-side channels: " << activeChannels_PSide << "/1024 (" << (100.0 * activeChannels_PSide / 1024) << "%)" << std::endl;
        summaryFile << "=== TIME INTERVAL ANALYSIS ===" << std::endl;
        summaryFile << "Global mean time interval: " << meanIntervalGlobal << " ± " << rmsIntervalGlobal << " ns" << std::endl;
        summaryFile << "N-side mean time interval: " << meanIntervalNSide << " ns" << std::endl;
        summaryFile << "P-side mean time interval: " << meanIntervalPSide << " ns" << std::endl;
        summaryFile << "Total time intervals recorded: " << gTimeIntervals.global_intervals.size() << std::endl;
        summaryFile << "N-side intervals: " << gTimeIntervals.nside_intervals.size() << std::endl;
        summaryFile << "P-side intervals: " << gTimeIntervals.pside_intervals.size() << std::endl;
        summaryFile.close();
    }
    
    std::cout << "Analysis completed for file: " << inputFile << std::endl;
    std::cout << "All plots saved to: " << plotsDir << std::endl;
}

// Add this function to analyze time intervals across all runs
void analyze_cross_run_time_intervals(const TString& outputDir) {
    TString plotsDir = outputDir + "/time_analysis_plots";
    gSystem->mkdir(plotsDir, kTRUE);
    
    // Combine all time intervals from all runs
    std::vector<double> allIntervals;
    std::vector<double> allNSideIntervals;
    std::vector<double> allPSideIntervals;
    
    for (const auto& run : gLastHitTime) {
        // You might want to add run-specific time analysis here
    }
    
    // Create comprehensive time analysis plots
    TH1F* hAllTimeIntervals = new TH1F("hAllTimeIntervals", 
        "Time intervals across all runs;Time Interval [ns];Count",
        200, 0, 1000);
    
    for (double interval : gTimeIntervals.global_intervals) {
        hAllTimeIntervals->Fill(interval);
    }
    
    TCanvas* cTimeSummary = new TCanvas("cTimeSummary", "Time Interval Summary", 800, 600);
    hAllTimeIntervals->Draw();
    cTimeSummary->SaveAs(Form("%s/all_runs_time_intervals.png", plotsDir.Data()));
    
    delete hAllTimeIntervals;
    delete cTimeSummary;
}

// Main analysis function
void analyze_digi_charge() {
    // Clear global data
    gLastHitTime.clear();
    gTimeIntervals.clear();
    gAnalysisResults.clear();
    
    // Set style
    gStyle->SetOptStat(1111);
    gStyle->SetOptFit(1);
    gStyle->SetPalette(55);
    
    // Read calibration data
    TString calibFile = "Calibration.txt";
    std::map<int, CalibrationData> calibrationMap = read_calibration_data(calibFile);
    
    // Create main output directory
    TString mainOutputDir = "charge_analysis_results";
    gSystem->mkdir(mainOutputDir, kTRUE);
    
    // Find all .digi.root files
    TSystemDirectory dir(".", ".");
    TList* files = dir.GetListOfFiles();
    std::vector<std::string> digiFiles;
    
    if (files) {
        TIter next(files);
        TSystemFile* file;
        
        while ((file = dynamic_cast<TSystemFile*>(next()))) {
            TString fname = file->GetName();
            if (fname.EndsWith(".digi.root")) {
                digiFiles.push_back(fname.Data());
                std::cout << "Found file: " << fname << std::endl;
            }
        }
        delete files;
    }
    
    if (digiFiles.empty()) {
        std::cout << "No .digi.root files found!" << std::endl;
        return;
    }
    
    std::sort(digiFiles.begin(), digiFiles.end());
    std::cout << "\nFound " << digiFiles.size() << " .digi.root files to analyze" << std::endl;
    
    // Analyze each file
    for (size_t i = 0; i < digiFiles.size(); i++) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Analyzing file " << i+1 << "/" << digiFiles.size() << ": " << digiFiles[i] << std::endl;
        std::cout << "========================================" << std::endl;
        
        analyze_single_file(digiFiles[i], mainOutputDir, calibrationMap);
    }
    
    // Create comprehensive analysis
    std::cout << "\nCreating comprehensive analysis..." << std::endl;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Analysis of all files completed!" << std::endl;
    std::cout << "Results are saved in: " << mainOutputDir << std::endl;
    std::cout << "========================================" << std::endl;
}

void run_charge_analysis() {
    analyze_digi_charge();
}