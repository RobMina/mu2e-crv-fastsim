// Resolve the capture geometry with SimpleConfig without constructing or transporting a G4 event.
#include <iostream>
#include <sstream>
#include <string>

#include "TTree.h"
#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Principal/Event.h"
#include "art_root_io/TFileService.h"
#include "cetlib_except/exception.h"
#include "fhiclcpp/types/Atom.h"
#include "Offline/ConfigTools/inc/SimpleConfig.hh"

namespace mu2e {

class CrvRecordingConfigDumper : public art::EDAnalyzer {
 public:
  struct Config {
    fhicl::Atom<std::string> inputFile{fhicl::Name("inputFile")};
    fhicl::Atom<bool> allowReplacement{fhicl::Name("allowReplacement"), true};
  };
  using Parameters = art::EDAnalyzer::Table<Config>;
  explicit CrvRecordingConfigDumper(const Parameters& conf) : art::EDAnalyzer(conf), conf_(conf()) {}
  void beginJob() override;
  void analyze(const art::Event&) override {}

 private:
  Config conf_;
  unsigned schema_{1};
  unsigned long long lines_{0}, hash_{0};
  int limit_{0}, explicit_{0}, replacement_{0}, default_{0};
  std::string input_, requested_, resolvedText_;
};

void CrvRecordingConfigDumper::beginJob() {
  SimpleConfig config(conf_.inputFile(), conf_.allowReplacement(), false, false);
  input_ = config.inputFile();
  requested_ = conf_.inputFile();
  std::ostringstream resolved;
  config.print(resolved);
  resolvedText_ = resolved.str();
  lines_ = config.inputFileLines();
  hash_ = config.inputFileHash();
  explicit_ = config.hasName("g4.stepsSizeLimit");
  limit_ = config.getInt("g4.stepsSizeLimit", 0);
  replacement_ = conf_.allowReplacement();
  art::ServiceHandle<art::TFileService> service;
  auto* tree = service->make<TTree>("config", "Readback resolution, to bind to original transport geometry hash");
  tree->Branch("schema_version", &schema_, "schema_version/i");
  tree->Branch("input_file", &input_);
  tree->Branch("requested_input_file", &requested_);
  tree->Branch("resolved_text", &resolvedText_);
  tree->Branch("steps_limit_default", &default_, "steps_limit_default/I");
  tree->Branch("input_lines", &lines_, "input_lines/l");
  tree->Branch("input_hash", &hash_, "input_hash/l");
  tree->Branch("steps_limit", &limit_, "steps_limit/I");
  tree->Branch("steps_limit_explicit", &explicit_, "steps_limit_explicit/I");
  tree->Branch("allow_replacement", &replacement_, "allow_replacement/I");
  if (tree->Fill() < 0) throw cet::exception("CrvRecordingConfigDumper") << "failed config tree write\n";
  config.printOpen(std::cout, "RawReadbackGeometry");
  std::cout << "RawReadbackGeometry g4.stepsSizeLimit=" << limit_ << " explicit=" << explicit_ << '\n';
  config.print(std::cout, "RawReadbackGeom: ");
}

}  // namespace mu2e

DEFINE_ART_MODULE(mu2e::CrvRecordingConfigDumper)
