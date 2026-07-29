"""
Raw Data Collection
This script processes the ATLAS Open Data dilepton dataset to 
extract events with exactly two electrons or two muons. 
For each event, it calculates the invariant mass of the lepton pair, 
counts the number of jets, identifies the leading jet's kinematics, 
and computes the missing transverse energy (MET). 
The results are saved to a CSV file for analysis in the BDT and DNN pipelines.
"""

# STEP 1 of the pipeline. Streams ATLAS Open Data over XRootD and writes one
# CSV of dilepton events per file range. Run this repeatedly to cover the
# dataset (edit START_FILE / END_FILE below), or set END_FILE = 10049 for all.
# EDIT BEFORE RUNNING: START_FILE, END_FILE, MAX_EVENTS (below).
# REQUIRES: pip install uproot awkward numpy pandas cernopendata-client
# OUTPUT: dilepton_events_files_<START>_to_<END-1>.csv  (feeds DNN.py / bdt)
# Ctrl+C saves a checkpoint; rerun to resume.


import subprocess
import os
import uproot
import awkward as ak
import numpy as np
import pandas as pd
import pickle
import signal
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Change working directory to script directory
os.chdir(script_dir)

# CONFIGURATION

START_FILE = 0        # EDIT: first file index to process (0 = first)
END_FILE   = 100      # EDIT: last file index (exclusive); full dataset is 10049 files
OUTPUT_CSV = f'dilepton_events_files_{START_FILE}_to_{END_FILE-1}.csv'  # Name includes file range
MAX_EVENTS = None     # EDIT: cap events (None = no cap)

# CHECKPOINT SYSTEM (For long runs, allows resuming after interruption without losing all progress)

CHECKPOINT_FILE = "event_processing_checkpoint.pkl"
CHECKPOINT_INTERVAL = 20  # Save checkpoint every N files
PAUSE_REQUESTED = False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully - save checkpoint and exit"""
    global PAUSE_REQUESTED
    print("\n\n" + "="*80)
    print("PAUSE REQUESTED - Saving checkpoint...")
    print("="*80)
    PAUSE_REQUESTED = True

signal.signal(signal.SIGINT, signal_handler)

def save_checkpoint(file_idx, event_data, failed_files):
    """Save current progress to checkpoint file"""
    checkpoint = {
        'last_completed_file': file_idx,
        'event_data': event_data,
        'failed_files': failed_files,
    }
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"✓ Checkpoint saved at file {file_idx + 1}")

def load_checkpoint():
    """Load checkpoint if it exists"""
    if os.path.exists(CHECKPOINT_FILE):
        print("="*80)
        print("CHECKPOINT FOUND!")
        print("="*80)
        response = input("Resume from checkpoint? (y/n): ").lower()
        if response == 'y':
            with open(CHECKPOINT_FILE, 'rb') as f:
                checkpoint = pickle.load(f)
            print(f"Resuming from file {checkpoint['last_completed_file'] + 2}")
            print("="*80 + "\n")
            return checkpoint
        else:
            print("Starting fresh analysis...")
            os.remove(CHECKPOINT_FILE)
    return None


DOI = "10.7483/OPENDATA.ATLAS.AOQL.8TT3" # DOI for the Open Data dataset

print(f"Fetching file URLs for DOI: {DOI}")
print("This may take a moment...\n")

try:
    result = subprocess.run(
        ['cernopendata-client', 'get-file-locations', '--doi', DOI, '--protocol', 'xrootd'],
        capture_output=True,
        text=True,
        check=True
    )
    
    file_urls = [url.strip() for url in result.stdout.strip().split('\n') if url.strip()]
    print(f"Found {len(file_urls)} total files")
    
except subprocess.CalledProcessError as e:
    print(f"Error running cernopendata-client: {e}")
    exit(1)
except FileNotFoundError:
    print("cernopendata-client not found!")
    print("Install it with: pip install cernopendata-client")
    exit(1)

# Apply file range limits
if END_FILE > len(file_urls):
    END_FILE = len(file_urls)
    print(f"Note: END_FILE adjusted to {END_FILE} (total available files)")

file_urls = file_urls[START_FILE:END_FILE]

print(f"\nProcessing files {START_FILE} to {END_FILE-1} ({len(file_urls)} files total)\n")
print("="*80)


# Try to load checkpoint
checkpoint = load_checkpoint()

if checkpoint:
    event_data = checkpoint['event_data']
    failed_files = checkpoint['failed_files']
    start_idx = checkpoint['last_completed_file'] + 1
    print(f"Loaded {len(event_data)} events from checkpoint")
    print(f"Previously failed files: {len(failed_files)}")
else:
    event_data = []
    failed_files = []
    start_idx = 0

for file_idx in range(start_idx, len(file_urls)):
    if MAX_EVENTS is not None and len(event_data) >= MAX_EVENTS:
        print(f"\n{'='*80}")
        print(f"REACHED EVENT LIMIT: {len(event_data)} events collected")
        print(f"{'='*80}")
        break
    
    if PAUSE_REQUESTED:
        save_checkpoint(file_idx - 1, event_data, failed_files)
        print("\n" + "="*80)
        print("PAUSING - Checkpoint saved")
        print("="*80)
        print("To resume: Just run this script again!")
        print(f"Progress: {file_idx}/{len(file_urls)} files completed")
        sys.exit(0)
    
    file_url = file_urls[file_idx]
    global_file_idx = START_FILE + file_idx
    
    print(f"\n{'='*80}")
    print(f"Processing file {file_idx + 1}/{len(file_urls)} (Global index: {global_file_idx}) || URL: {file_url}")
    print(f"{'='*80}\n")
    
    try:
        file = uproot.open(file_url)
        tree = file["CollectionTree"]
        
        analysis_electrons = {
            'pt': tree["AnalysisElectronsAuxDyn.pt"].array(),
            'eta': tree["AnalysisElectronsAuxDyn.eta"].array(),
            'phi': tree["AnalysisElectronsAuxDyn.phi"].array(),
        }
        
        analysis_muons = {
            'pt': tree["AnalysisMuonsAuxDyn.pt"].array(),
            'eta': tree["AnalysisMuonsAuxDyn.eta"].array(),
            'phi': tree["AnalysisMuonsAuxDyn.phi"].array(),
        }
        
        analysis_jets = {
            'pt': tree["AnalysisJetsAuxDyn.pt"].array(),
            'eta': tree["AnalysisJetsAuxDyn.eta"].array(),
            'phi': tree["AnalysisJetsAuxDyn.phi"].array(),
        }
        
        try:
            met_mpx = tree["MET_Core_AnalysisMETAuxDyn.mpx"].array()
            met_mpy = tree["MET_Core_AnalysisMETAuxDyn.mpy"].array()
        except Exception as e:
            print(f"  Warning: Could not load MET - {e}")
            met_mpx = None
            met_mpy = None
        
        mask_2e = ak.num(analysis_electrons['pt']) == 2
        mask_2mu = ak.num(analysis_muons['pt']) == 2
        
        n_dielectron = np.sum(mask_2e)
        n_dimuon = np.sum(mask_2mu)
        
        print(f"  Events with 2 electrons: {n_dielectron}")
        print(f"  Events with 2 muons: {n_dimuon}")
        
        for i in range(len(mask_2e)):
            if mask_2e[i]:
                pt1 = float(analysis_electrons['pt'][i][0]) / 1000.0
                eta1 = float(analysis_electrons['eta'][i][0])
                phi1 = float(analysis_electrons['phi'][i][0])
                
                pt2 = float(analysis_electrons['pt'][i][1]) / 1000.0
                eta2 = float(analysis_electrons['eta'][i][1])
                phi2 = float(analysis_electrons['phi'][i][1])
                
                invariant_mass = np.sqrt(
                    2 * pt1 * pt2 * 
                    (np.cosh(eta1 - eta2) - np.cos(phi1 - phi2))
                )
                
                jets_in_event = analysis_jets['pt'][i]
                n_jets = len(jets_in_event)
                if n_jets > 0:
                    jet_pts = [float(pt) for pt in jets_in_event]
                    leading_idx = jet_pts.index(max(jet_pts))
                    leading_jet_pt = jet_pts[leading_idx] / 1000.0
                    leading_jet_eta = float(analysis_jets['eta'][i][leading_idx])
                    leading_jet_phi = float(analysis_jets['phi'][i][leading_idx])
                else:
                    leading_jet_pt = 0.0
                    leading_jet_eta = 0.0
                    leading_jet_phi = 0.0
                
                if met_mpx is not None:
                    mpx = float(met_mpx[i][0]) if len(met_mpx[i]) > 0 else 0.0
                    mpy = float(met_mpy[i][0]) if len(met_mpy[i]) > 0 else 0.0
                    met = np.sqrt(mpx**2 + mpy**2) / 1000.0
                else:
                    met = 0.0
                
                event_data.append({
                    'file_idx': global_file_idx,
                    'type': 'e',
                    'pt1': pt1,
                    'eta1': eta1,
                    'phi1': phi1,
                    'pt2': pt2,
                    'eta2': eta2,
                    'phi2': phi2,
                    'invariant_mass': invariant_mass,
                    'n_jets': n_jets,
                    'leading_jet_pt': leading_jet_pt,
                    'leading_jet_eta': leading_jet_eta,
                    'leading_jet_phi': leading_jet_phi,
                    'met': met
                })
        
        for i in range(len(mask_2mu)):
            if mask_2mu[i]:
                pt1 = float(analysis_muons['pt'][i][0]) / 1000.0
                eta1 = float(analysis_muons['eta'][i][0])
                phi1 = float(analysis_muons['phi'][i][0])
                
                pt2 = float(analysis_muons['pt'][i][1]) / 1000.0
                eta2 = float(analysis_muons['eta'][i][1])
                phi2 = float(analysis_muons['phi'][i][1])
                
                invariant_mass = np.sqrt(
                    2 * pt1 * pt2 * 
                    (np.cosh(eta1 - eta2) - np.cos(phi1 - phi2))
                )
                
                jets_in_event = analysis_jets['pt'][i]
                n_jets = len(jets_in_event)
                if n_jets > 0:
                    jet_pts = [float(pt) for pt in jets_in_event]
                    leading_idx = jet_pts.index(max(jet_pts))
                    leading_jet_pt = jet_pts[leading_idx] / 1000.0
                    leading_jet_eta = float(analysis_jets['eta'][i][leading_idx])
                    leading_jet_phi = float(analysis_jets['phi'][i][leading_idx])
                else:
                    leading_jet_pt = 0.0
                    leading_jet_eta = 0.0
                    leading_jet_phi = 0.0
                
                if met_mpx is not None:
                    mpx = float(met_mpx[i][0]) if len(met_mpx[i]) > 0 else 0.0
                    mpy = float(met_mpy[i][0]) if len(met_mpy[i]) > 0 else 0.0
                    met = np.sqrt(mpx**2 + mpy**2) / 1000.0
                else:
                    met = 0.0
                
                event_data.append({
                    'file_idx': global_file_idx,
                    'type': 'm',
                    'pt1': pt1,
                    'eta1': eta1,
                    'phi1': phi1,
                    'pt2': pt2,
                    'eta2': eta2,
                    'phi2': phi2,
                    'invariant_mass': invariant_mass,
                    'n_jets': n_jets,
                    'leading_jet_pt': leading_jet_pt,
                    'leading_jet_eta': leading_jet_eta,
                    'leading_jet_phi': leading_jet_phi,
                    'met': met
                })
        
        print(f"  ✓ Collected {n_dielectron + n_dimuon} events from this file")
        print(f"  Total events so far: {len(event_data)}")
        
        # Save checkpoint periodically
        if (file_idx + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(file_idx, event_data, failed_files)
        
        del tree, file
        
    except Exception as e:
        print(f"  ERROR processing file {file_idx + 1}: {e}")
        failed_files.append({
            'file_idx': global_file_idx,
            'file_url': file_url,
            'error': str(e)
        })
        print(f"  ✗ Added to failed files list")
        continue


print(f"\n{'='*80}")
print("SAVING RESULTS")
print(f"{'='*80}\n")

df = pd.DataFrame(event_data)

df.to_csv(OUTPUT_CSV, index=False)

print(f"✓ Saved {len(df)} events to {OUTPUT_CSV}")
print(f"  - {len(df[df['type'] == 'e'])} dielectron events")
print(f"  - {len(df[df['type'] == 'm'])} dimuon events")
print(f"  - From files {START_FILE} to {END_FILE-1}")

if failed_files:
    failed_df = pd.DataFrame(failed_files)
    failed_csv = f'failed_files_{START_FILE}_to_{END_FILE-1}.csv'
    failed_df.to_csv(failed_csv, index=False)
    print(f"\n✗ {len(failed_files)} files failed - saved to {failed_csv}")
    print("\nFailed files:")
    for fail in failed_files[:5]:
        print(f"  File {fail['file_idx']}: {fail['error']}")
    if len(failed_files) > 5:
        print(f"  ... and {len(failed_files) - 5} more (see {failed_csv})")
else:
    print(f"\n✓ All files processed successfully!")

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"\nDielectron events:")
print(df[df['type'] == 'e'][['invariant_mass', 'met', 'n_jets']].describe())
print(f"\nDimuon events:")
print(df[df['type'] == 'm'][['invariant_mass', 'met', 'n_jets']].describe())

# Clean up checkpoint file on successful completion
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print("\n✓ Checkpoint file cleaned up - processing completed successfully!")

print("\n" + "="*80)
print("PROCESSING COMPLETE!")
print("="*80)