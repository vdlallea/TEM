# Microstrip R and L Solver

Volume Integral Equation-solver voor de per-lengte-eenheid weerstand R(f) en
inductie L(f) van een koperen microstrip-lijn.

## Installatie
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

## Gebruik
python scripts/run_all.py

## Structuur hopelijk: 
TEM/
    .gitignore          # al aanwezig via GitHub
    README.md
    requirements.txt
    src/
        __init__.py
        geometry.py     # (later) discretisatie
        kernels.py      # (later) log-kern en Appendix B-integralen
        solver.py       # (later) opbouw R, L en oplossing
        plotting.py     # (later) figuren
    tests/              # (later) validatietests
    scripts/
        run_all.py      # (later) maakt alle figuren uit het rapport
    results/            # gegenereerde figuren en data
    report/             # rapport (Word + PDF)

## Auteurs
Groep 5: Conings Noah, Delalleau Vince, Van Wezemael Gus