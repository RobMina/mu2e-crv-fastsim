// Saved-raw diagnostic schema and audit boundaries are defined in docs/native-raw-integrity.md.
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <string>

#include "TTree.h"
#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "art_root_io/TFileService.h"
#include "canvas/Persistency/Common/Ptr.h"
#include "canvas/Utilities/InputTag.h"
#include "cetlib_except/exception.h"
#include "fhiclcpp/types/Atom.h"

#include "Offline/MCDataProducts/inc/GenParticle.hh"
#include "Offline/MCDataProducts/inc/SimParticle.hh"
#include "Offline/MCDataProducts/inc/StatusG4.hh"
#include "Offline/MCDataProducts/inc/StepPointMC.hh"

namespace mu2e {
namespace {
using U32 = std::uint32_t;
using U64 = std::uint64_t;
using I32 = std::int32_t;

template <typename T>
void scalar(TTree* tree, const std::string& name, T& value, const char type) {
  const std::string leaf = name + "/" + type;
  tree->Branch(name.c_str(), &value, leaf.c_str());
}

struct PointerDiagnostic {
  I32 nonnull{0};
  U32 product_id{0};
  U64 key{0};
  I32 available{0}, dereferenced{0}, member{0}, matches_selected{0}, error{0};

  void branches(TTree* tree, const std::string& prefix) {
    scalar(tree, prefix + "_nonnull", nonnull, 'I');
    scalar(tree, prefix + "_product_id", product_id, 'i');
    scalar(tree, prefix + "_key", key, 'l');
    scalar(tree, prefix + "_available", available, 'I');
    scalar(tree, prefix + "_dereferenced", dereferenced, 'I');
    scalar(tree, prefix + "_member", member, 'I');
    scalar(tree, prefix + "_matches_selected", matches_selected, 'I');
    scalar(tree, prefix + "_error", error, 'I');
  }
};

template <typename T>
using SelectedElements = std::map<U64, const T*>;

template <typename T>
PointerDiagnostic inspectPointer(const art::Ptr<T>& ptr, const I32 selectedPresent,
                                 const U32 selectedID, const SelectedElements<T>& elements) {
  PointerDiagnostic result;
  try {
    result.nonnull = ptr.isNonnull();
    result.product_id = ptr.id().value();
    if (!result.nonnull) return result;
    result.key = ptr.key();
    const auto found = elements.find(result.key);
    result.member = selectedPresent && result.product_id == selectedID && found != elements.end();
    result.available = ptr.isAvailable();
    // Resolution is attempted only after both membership and availability have been established.
    if (result.member && result.available) {
      const T* resolved = ptr.get();
      result.dereferenced = resolved != nullptr;
      result.matches_selected = result.dereferenced && resolved == found->second;
    }
  } catch (const std::exception&) {
    result.error = 1;
  } catch (...) {
    result.error = 1;
  }
  return result;
}

template <typename T>
const T* readProduct(const art::Event& event, const art::InputTag& tag, art::Handle<T>& handle,
                     I32& present, U32& productID, I32& collectionError) {
  present = 0;
  productID = 0;
  try {
    event.getByLabel(tag, handle);
    if (handle.isValid()) {
      const T* product = handle.product();
      if (product != nullptr) {
        productID = handle.id().value();
        present = 1;
        return product;
      }
    }
  } catch (const std::exception&) {
    collectionError = 1;
  } catch (...) {
    collectionError = 1;
  }
  collectionError = 1;
  return nullptr;
}

void fill(TTree* tree) {
  if (tree->Fill() < 0)
    throw cet::exception("CrvRawIntegrityDumper") << "Failed to fill " << tree->GetName() << '\n';
}
}  // namespace

class CrvRawIntegrityDumper : public art::EDAnalyzer {
 public:
  struct Config {
    fhicl::Atom<art::InputTag> genParticles{fhicl::Name("genParticles")};
    fhicl::Atom<art::InputTag> simParticles{fhicl::Name("simParticles")};
    fhicl::Atom<art::InputTag> rawSteps{fhicl::Name("rawSteps")};
    fhicl::Atom<art::InputTag> status{fhicl::Name("status")};
  };
  using Parameters = art::EDAnalyzer::Table<Config>;
  explicit CrvRawIntegrityDumper(const Parameters& conf);
  void beginJob() override;
  void analyze(const art::Event& event) override;

 private:
  Config conf_;
  TTree *events_{nullptr}, *gen_{nullptr}, *particles_{nullptr}, *raw_{nullptr}, *daughters_{nullptr};
  U32 run_{0}, subrun_{0}, event_{0}, schemaVersion_{1};
  U32 genProduct_{0}, simProduct_{0}, rawProduct_{0}, statusProduct_{0};
  I32 genPresent_{0}, simPresent_{0}, rawPresent_{0}, statusPresent_{0}, collectionError_{0};
  U64 nGen_{0}, nParticles_{0}, nRaw_{0}, nDaughters_{0};
  I32 status_{-1}, nG4Tracks_{0}, overflowSimP_{0}, nStepLimit_{0}, nFieldPropagator_{0};
  double cpuTime_{0}, realTime_{0};
  std::string genTag_, simTag_, rawTag_, statusTag_;
  U64 genIdx_{0};
  I32 genPdg_{0}, genId_{0};
  double e_{0}, t_{0}, x_{0}, y_{0}, z_{0}, px_{0}, py_{0}, pz_{0};
  U64 particleKey_{0}, particleID_{0}, particleNDaughters_{0};
  I32 particlePdg_{0}, isPrimary_{0};
  PointerDiagnostic parentPointer_, genPointer_;
  U64 rawIdx_{0};
  U32 bar_{0};
  PointerDiagnostic simPointer_;
  U64 ownerKey_{0}, daughterIdx_{0};
  PointerDiagnostic daughterPointer_;
};

CrvRawIntegrityDumper::CrvRawIntegrityDumper(const Parameters& conf)
    : art::EDAnalyzer(conf), conf_(conf()) {
  genTag_ = conf_.genParticles().encode();
  simTag_ = conf_.simParticles().encode();
  rawTag_ = conf_.rawSteps().encode();
  statusTag_ = conf_.status().encode();
  if (genTag_ != "cosmicTimeOffset::Primary" || simTag_ != "g4run::R3Inclusive" ||
      rawTag_ != "g4run:CRV:R3Inclusive" || statusTag_ != "g4run::R3Inclusive")
    throw cet::exception("CrvRawIntegrityDumper")
        << "Schema v1 requires genParticles=cosmicTimeOffset::Primary, simParticles=g4run::R3Inclusive, "
        << "rawSteps=g4run:CRV:R3Inclusive, status=g4run::R3Inclusive\n";
}

void CrvRawIntegrityDumper::beginJob() {
  art::ServiceHandle<art::TFileService> tfs;
  const auto makeTree = [this, &tfs](const char* name, const char* title) {
    TTree* tree = tfs->make<TTree>(name, title);
    scalar(tree, "run", run_, 'i');
    scalar(tree, "subrun", subrun_, 'i');
    scalar(tree, "event", event_, 'i');
    return tree;
  };
  events_ = makeTree("events", "Saved-raw schema v1 all-event census and collection diagnostics");
  scalar(events_, "schema_version", schemaVersion_, 'i');
  scalar(events_, "gen_product_id", genProduct_, 'i');
  scalar(events_, "sim_product_id", simProduct_, 'i');
  scalar(events_, "raw_product_id", rawProduct_, 'i');
  scalar(events_, "status_product_id", statusProduct_, 'i');
  scalar(events_, "gen_present", genPresent_, 'I');
  scalar(events_, "sim_present", simPresent_, 'I');
  scalar(events_, "raw_present", rawPresent_, 'I');
  scalar(events_, "status_present", statusPresent_, 'I');
  scalar(events_, "collection_error", collectionError_, 'I');
  scalar(events_, "n_gen", nGen_, 'l');
  scalar(events_, "n_particles", nParticles_, 'l');
  scalar(events_, "n_raw", nRaw_, 'l');
  scalar(events_, "n_daughters", nDaughters_, 'l');
  scalar(events_, "status", status_, 'I');
  scalar(events_, "nG4Tracks", nG4Tracks_, 'I');
  scalar(events_, "overflowSimP", overflowSimP_, 'I');
  scalar(events_, "nStepLimit", nStepLimit_, 'I');
  scalar(events_, "nFieldPropagator", nFieldPropagator_, 'I');
  scalar(events_, "cpuTime", cpuTime_, 'D');
  scalar(events_, "realTime", realTime_, 'D');
  events_->Branch("gen_tag", &genTag_);
  events_->Branch("sim_tag", &simTag_);
  events_->Branch("raw_tag", &rawTag_);
  events_->Branch("status_tag", &statusTag_);

  gen_ = makeTree("gen", "Every selected original GenParticle");
  scalar(gen_, "idx", genIdx_, 'l');
  scalar(gen_, "gen_product_id", genProduct_, 'i');
  scalar(gen_, "pdg", genPdg_, 'I');
  scalar(gen_, "genid", genId_, 'I');
  scalar(gen_, "e", e_, 'D');
  scalar(gen_, "t", t_, 'D');
  scalar(gen_, "x", x_, 'D');
  scalar(gen_, "y", y_, 'D');
  scalar(gen_, "z", z_, 'D');
  scalar(gen_, "px", px_, 'D');
  scalar(gen_, "py", py_, 'D');
  scalar(gen_, "pz", pz_, 'D');

  particles_ = makeTree("particles", "Every selected SimParticle and serialized associations");
  scalar(particles_, "key", particleKey_, 'l');
  scalar(particles_, "id", particleID_, 'l');
  scalar(particles_, "n_daughters", particleNDaughters_, 'l');
  scalar(particles_, "sim_product_id", simProduct_, 'i');
  scalar(particles_, "pdg", particlePdg_, 'I');
  scalar(particles_, "isprim", isPrimary_, 'I');
  parentPointer_.branches(particles_, "parent");
  genPointer_.branches(particles_, "gen");

  raw_ = makeTree("raw", "Every saved raw CRV StepPointMC association");
  scalar(raw_, "idx", rawIdx_, 'l');
  scalar(raw_, "raw_product_id", rawProduct_, 'i');
  scalar(raw_, "bar", bar_, 'i');
  simPointer_.branches(raw_, "sim");

  daughters_ = makeTree("daughters", "Every serialized daughter association in owner order");
  scalar(daughters_, "owner_key", ownerKey_, 'l');
  scalar(daughters_, "idx", daughterIdx_, 'l');
  scalar(daughters_, "sim_product_id", simProduct_, 'i');
  daughterPointer_.branches(daughters_, "daughter");
}

void CrvRawIntegrityDumper::analyze(const art::Event& event) {
  run_ = event.run();
  subrun_ = event.subRun();
  event_ = event.event();
  collectionError_ = 0;
  nGen_ = nParticles_ = nRaw_ = nDaughters_ = 0;
  status_ = -1;
  nG4Tracks_ = overflowSimP_ = nStepLimit_ = nFieldPropagator_ = 0;
  cpuTime_ = realTime_ = 0;

  art::Handle<GenParticleCollection> gh;
  art::Handle<SimParticleCollection> ph;
  art::Handle<StepPointMCCollection> rh;
  art::Handle<StatusG4> sh;
  const auto* gens = readProduct(event, conf_.genParticles(), gh, genPresent_, genProduct_, collectionError_);
  const auto* parts = readProduct(event, conf_.simParticles(), ph, simPresent_, simProduct_, collectionError_);
  const auto* hits = readProduct(event, conf_.rawSteps(), rh, rawPresent_, rawProduct_, collectionError_);
  const auto* status = readProduct(event, conf_.status(), sh, statusPresent_, statusProduct_, collectionError_);

  if (status != nullptr) {
    status_ = status->status();
    nG4Tracks_ = status->nG4Tracks();
    overflowSimP_ = status->overflowSimParticles();
    nStepLimit_ = status->nKilledStepLimit();
    nFieldPropagator_ = status->nKilledByFieldPropagator();
    cpuTime_ = status->cpuTime();
    realTime_ = status->realTime();
  }

  SelectedElements<GenParticle> selectedGen;
  SelectedElements<SimParticle> selectedSim;
  if (gens != nullptr) {
    nGen_ = gens->size();
    U64 idx = 0;
    for (const auto& g : *gens) selectedGen.emplace(idx++, std::addressof(g));
  }
  if (parts != nullptr) {
    nParticles_ = parts->size();
    for (const auto& entry : *parts) {
      selectedSim.emplace(entry.first.asUint(), std::addressof(entry.second));
      nDaughters_ += entry.second.daughters().size();
    }
  }
  if (hits != nullptr) nRaw_ = hits->size();

  if (gens != nullptr) {
    genIdx_ = 0;
    for (const auto& g : *gens) {
      genPdg_ = g.pdgId();
      genId_ = g.generatorId().id();
      e_ = g.momentum().e();
      t_ = g.time();
      x_ = g.position().x();
      y_ = g.position().y();
      z_ = g.position().z();
      px_ = g.momentum().px();
      py_ = g.momentum().py();
      pz_ = g.momentum().pz();
      fill(gen_);
      ++genIdx_;
    }
  }
  if (parts != nullptr) {
    for (const auto& entry : *parts) {
      const auto& p = entry.second;
      particleKey_ = entry.first.asUint();
      particleID_ = p.id().asUint();
      particleNDaughters_ = p.daughters().size();
      particlePdg_ = p.pdgId();
      parentPointer_ = inspectPointer(p.parent(), simPresent_, simProduct_, selectedSim);
      genPointer_ = inspectPointer(p.genParticle(), genPresent_, genProduct_, selectedGen);
      // This is the pinned isPrimary definition without a second unguarded pointer query.
      isPrimary_ = genPointer_.nonnull && p.creationCode() == ProcessCode::mu2ePrimary;
      fill(particles_);
      ownerKey_ = particleKey_;
      daughterIdx_ = 0;
      for (const auto& daughter : p.daughters()) {
        daughterPointer_ = inspectPointer(daughter, simPresent_, simProduct_, selectedSim);
        fill(daughters_);
        ++daughterIdx_;
      }
    }
  }
  if (hits != nullptr) {
    rawIdx_ = 0;
    for (const auto& hit : *hits) {
      bar_ = hit.barIndex().asUint();
      simPointer_ = inspectPointer(hit.simParticle(), simPresent_, simProduct_, selectedSim);
      fill(raw_);
      ++rawIdx_;
    }
  }
  fill(events_);
}

}  // namespace mu2e

DEFINE_ART_MODULE(mu2e::CrvRawIntegrityDumper)
