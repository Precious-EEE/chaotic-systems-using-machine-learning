"""
=============================================================
  RUN ALL — Master Script
=============================================================
Run this single file to execute the entire project pipeline:

  python run_all.py

Or run individual steps:
  python 01_lorenz_system.py
  python 02_reservoir_computing.py
  python 03_neural_ode.py
  python 04_comparison_analysis.py
  python 05_summary_report.py

Estimated total runtime on a laptop (CPU):
  Step 1 — ~1 min
  Step 2 — ~2 min
  Step 3 — ~10-20 min  (Neural ODE training)
  Step 4 — ~3 min
  Step 5 — ~1 min
  Total  — ~20-30 min
"""

import subprocess, sys, time

STEPS = [
    ("01_lorenz_system.py",      "Lorenz System & Lyapunov Exponents"),
    ("02_reservoir_computing.py","Echo State Network (Reservoir Computing)"),
    ("03_neural_ode.py",         "Neural ODE Training"),
    ("04_comparison_analysis.py","Model Comparison & Attractor Analysis"),
    ("05_summary_report.py",     "Summary Report & Final Figure"),
]

def run_step(script, name):
    print(f"\n{'='*65}")
    print(f"  RUNNING: {name}")
    print(f"  Script:  {script}")
    print(f"{'='*65}\n")
    t0 = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  ❌ {script} failed (exit code {result.returncode})")
        return False
    print(f"\n  ✅ Completed in {elapsed/60:.1f} min")
    return True

if __name__ == "__main__":
    print("╔" + "═"*63 + "╗")
    print("║  CHAOS × ML PROJECT — Full Pipeline".center(65) + "║")
    print("║  Neural ODEs & Reservoir Computing for Chaotic Systems".center(65) + "║")
    print("╚" + "═"*63 + "╝")

    total_start = time.time()
    for script, name in STEPS:
        ok = run_step(script, name)
        if not ok:
            print(f"\nPipeline stopped at: {script}")
            print("Fix the error above and re-run: python run_all.py")
            sys.exit(1)

    total = (time.time() - total_start) / 60
    print(f"\n🎉  All steps complete in {total:.1f} min!")
    print("   Check the plots/ folder for all visualisations.")
    print("   Check results/ for saved model outputs and metrics.")