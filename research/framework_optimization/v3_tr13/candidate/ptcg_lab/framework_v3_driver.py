"""Finite-engineering components; does not invoke the stopped TR13 run loop."""
from .framework_v2_training import NativeCheckpoints
from .training_control import TrainingControl
from .archive_outbox import ArchiveOutbox
from .framework_v3_runtime import from_config,ProgressPublisher
from .io import atomic_json,read_json

def components(config,output,epoch,profile,deadline,*,capacity_root,hard_bytes,progress):
    switches=from_config(config,output)
    if not switches:raise ValueError('Explicit admitted v3 engineering scope required')
    retained=config.get('readonly_retained_bytes');ticket=read_json(output/'ticket.json')
    if type(retained) is not int or retained<0 or retained!=ticket.get('readonly_retained_bytes'):raise ValueError('Readonly HDD1 bytes require original controller inventory/ticket binding')
    if type(hard_bytes) is not int or not 0<hard_bytes<=80*2**30:raise ValueError('Unchanged server total working-set ceiling')
    backend=NativeCheckpoints(output,epoch,profile,capacity_root=capacity_root,hard_bytes=hard_bytes,retained_bytes=retained,optimizations=switches)
    if switches['capacity_transaction']:
        import os
        from .framework_v3_admission import install
        install(backend,os.environ['PTCG_PROJECT_ROOT'])
    control=TrainingControl(output,output.name,epoch,deadline,progress,optimizations=switches)
    outbox=ArchiveOutbox(output,output.name,epoch,optimizations=switches)
    publisher=ProgressPublisher(atomic_json) if switches['progress_coalescing'] else atomic_json
    return backend,control,outbox,publisher
