"""Wait for consumed data durability and rotate both recovery references."""
import time
from pathlib import Path
from .io import read_json,atomic_json,utcnow
from .archive_prefix import canonical_hash
from .archive_ack import safe_target
from .training_control import ControlledTrainingStop
from .phase_timing import timed,span


def index_metadata(outbox,cursor):
    value=read_json(outbox.root/'checkpoint-commits/index.json');points=value['commits']
    if value.get('run_id')!=outbox.run_id or value.get('lease_epoch')!=outbox.epoch or not 1<=len(points)<=2:
        raise ValueError('Archive barrier recovery owner differs')
    if points[-1]['cursor']!=cursor:raise ValueError('Model cursor changed during archive barrier')
    for point in points:
        if canonical_hash({k:v for k,v in point.items() if k!='commit_id'})!=point['commit_id']:
            raise ValueError('Archive barrier commit metadata changed')
    return points


@timed('consumed_archive_barrier')
def retire_before_sampling(outbox,consumed,cursor,save,deadline,reserve,progress,control=None):
    """Called after the full new model is saved, before another rollout.

    Metadata decides whether another writer save is needed. The save callback
    still performs every production full hash, publication, two-point pin and
    ACK-gated unlink. This function never edits an index or deletes a file.
    """
    targets={r['path']:r for r in consumed}
    if len(targets)!=len(consumed):raise ValueError('Duplicate consumed barrier reference')
    rows={r['object']:r for r in outbox.records()}
    for name,ref in targets.items():
        row=rows.get(name)
        if row is None or (row['sha256'],row['bytes'])!=(ref['sha256'],ref['bytes']):
            raise ValueError('Consumed BO3 lacks its exact queued immutable original')
    def check():
        if control:control.wait(reserve(),gpu_required=False)
        if time.time()>deadline-reserve():raise ControlledTrainingStop('Consumed archive barrier reached unchanged checkpoint deadline')
    def pinned():
        found=set()
        for point in index_metadata(outbox,cursor):
            for ref in point['pending']:
                if ref['path'] in targets:
                    if ref!=targets[ref['path']]:raise ValueError('Consumed recovery reference changed')
                    found.add(ref['path'])
        return found
    initial_pins=len(pinned());started=time.monotonic();samples=0;last=0
    while True:
        check();pending=[r for name,r in rows.items() if name in targets and outbox._ack(r) is None]
        if not pending:break
        samples+=1;now=time.monotonic()
        if last==0 or now-last>=10:
            progress('waiting_consumed_archive_ACK',pending_consumed_objects=len(pending),
                pending_consumed_bytes=sum(r['bytes'] for r in pending),new_rollouts_blocked=True)
            last=now
        with span('archive_ACK_wait'):time.sleep(1)
    ack_wait_seconds=time.monotonic()-started;rotations=0
    while pinned():
        check()
        if rotations>=2:raise RuntimeError('Two post-ACK saves did not clear consumed recovery references')
        save();rotations+=1
    if len(index_metadata(outbox,cursor))!=2:raise ValueError('Both recovery points must be retained')
    if any(safe_target(outbox.root,name).exists() for name in targets):
        raise RuntimeError('ACKed consumed replicas still occupy the rollout workset')
    result={'at':utcnow(),'passed':True,'cursor':cursor,'consumed_files':len(targets),
        'consumed_logical_bytes':sum(r['bytes'] for r in consumed),'initial_pinned_consumed_files':initial_pins,
        'ACK_wait_samples':samples,'ACK_wait_seconds':ack_wait_seconds,'total_barrier_seconds':time.monotonic()-started,
        'post_ACK_writer_rotations':rotations,'both_retained_points_clear_of_consumed_refs':True,
        'all_consumed_originals_durable_before_own_replica_retirement':True,'new_rollouts_during_barrier':0}
    path=outbox.root/'archive-retirement-barriers.json';history=read_json(path) if path.exists() else []
    if len(history)>=256:raise ValueError('Bounded archive barrier journal exhausted')
    atomic_json(path,[*history,result]);progress('consumed_archive_barrier_passed',**{k:v for k,v in result.items() if k!='at'})
    return result
