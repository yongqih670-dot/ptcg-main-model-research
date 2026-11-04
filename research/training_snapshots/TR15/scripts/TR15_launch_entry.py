from pathlib import Path
import os,sys
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
D=R/'research/autonomy/activations/TR15-selfplay-100000-20261101'
with (D/'finite-controller.log').open('x',encoding='utf-8',buffering=1) as log:
    sys.stdout=log;sys.stderr=log
    import TR15_detached_executor
    TR15_detached_executor.main()
