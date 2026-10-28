"""Read-only unique PPO accounting across canonical experiment families."""
import copy
import hashlib
import io
import os
from pathlib import Path
import time

from .io import atomic_json, read_json, sha256, utcnow
from .ppo_accounting import ACCOUNTING, checkpoint_decisions, update_decisions


def run(config, output):
    import torch
    from .archive_compressed import verify_archive
    from .evidence_tiering import read_missing
    root=Path(config['project_path']).resolve();output=Path(output)
    deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX'])
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name or not 0<deadline-time.time()<=180.5:
        raise ValueError('One controlled180s read-only volume audit required')
    oldref=config['legacy_ledger'];oldpath=(root/oldref['path']).resolve()
    if root not in oldpath.parents or sha256(oldpath)!=oldref['sha256']:
        raise ValueError('Legacy scale ledger changed')
    old=read_json(oldpath);families=[];details=[]
    for ref in config.get('input_evidence',[]):
        path=(root/ref['path']).resolve()
        if root not in path.parents or sha256(path)!=ref['sha256']:raise ValueError('Bound source evidence changed')
    existing={row['hypothesis_id']:row for row in old['families']}
    if (len(existing)!=len(old['families']) or len(config['families'])!=len(existing)+1
            or set(existing)|{'TR10'}!={r['hypothesis_id'] for r in config['families']}):
        raise ValueError('Missing or duplicated canonical experiment family')
    for item in config['families']:
        if time.time()>deadline-15:raise TimeoutError('Original volume audit deadline')
        identity=item['hypothesis_id'];kind=item['kind'];payload=None
        if kind=='zero':
            if identity!='TR06' or item['expected_consumed_decisions']!=0 or item['expected_updates']!=0:
                raise ValueError('Only original closed zero-update TR06 has no optimizer trace')
            rows=[];consumed=unique=0;version=0
        elif kind=='trace':
            path=(root/item['path']).resolve()
            if root not in path.parents or sha256(path)!=item['sha256']:raise ValueError('Immutable update trace changed')
            rows=read_json(path);version=len(rows)
            consumed=sum(r['decisions'] for r in rows)
            unique=checkpoint_decisions(dict(version=version,trained_decisions=consumed,updates=rows))
        else:
            if kind=='file_checkpoint':
                path=(root/item['path']).resolve()
                if root not in path.parents:raise ValueError('Checkpoint outside project')
                payload=path.read_bytes() if path.exists() else read_missing(path)
                if payload is None:raise ValueError('Exact original checkpoint unavailable')
            elif kind=='online_checkpoint':
                path=(root/item['ACK']['path']).resolve()
                if root not in path.parents or sha256(path)!=item['ACK']['sha256']:raise ValueError('Exact checkpoint ACK changed')
                ack=read_json(path);archive=(root/item['archive_root']).resolve()
                if root not in archive.parents or ack['sha256']!=item['sha256']:raise ValueError('Checkpoint archive identity differs')
                payload=verify_archive(archive,ack,collect=True)
            else:raise ValueError('Unknown immutable volume evidence')
            if hashlib.sha256(payload).hexdigest()!=item['sha256']:raise ValueError('Original checkpoint SHA differs')
            saved=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
            if len(saved['model'])!=127 or len(saved['optimizer']['state'])!=97:raise ValueError('Incomplete committed state')
            rows=saved['updates'];consumed=saved['trained_decisions'];version=saved['version'];unique=checkpoint_decisions(saved)
            del saved,payload
        if (consumed,version)!=(item['expected_consumed_decisions'],item['expected_updates']):
            raise ValueError('Selected history differs from committed canonical family')
        if identity in existing and existing[identity]['trained_unique_decisions']!=consumed:
            raise ValueError('Legacy historical family was silently replaced')
        row=copy.deepcopy(existing.get(identity,dict(hypothesis_id=identity,
            canonical_training_family=config['TR10_canonical_family'],scientific_status='pending_original_evaluation')))
        row.update(legacy_consumed_decisions=consumed,trained_unique_decisions=unique,PPO_updates=version,
            unused_by_optimizer_decisions=consumed-unique,actual_counter_accounting=ACCOUNTING,
            actual_counter_source=copy.deepcopy(item))
        if identity in existing:
            row['training_target_completed_scope']='Preserved historical original declaration; not corrected actual-count coverage or permission to reopen.'
        else:
            row['forward_actual_target_decisions']=32768
            row['forward_actual_target_reached']=unique>=32768
        families.append(row)
        details.append(dict(hypothesis_id=identity,legacy_consumed_decisions=consumed,unique_actual_PPO_trained_decisions=unique,
            PPO_updates=version,per_update=[dict(version=r['version'],consumed=r['decisions'],
                actual_unique=update_decisions(r)) for r in rows]))
    proposal=copy.deepcopy(old);total=sum(r['trained_unique_decisions'] for r in families)
    proposal.update(at=utcnow(),scope='Canonical TR01-TR10 actual committed unique decisions proved by successful first-epoch optimizer coverage.',
        decision_accounting=ACCOUNTING,actual_unique_training_decisions=total,
        remaining_unique_training_decisions=max(0,proposal['goal_unique_actual_PPO_training_decisions']-total),
        families=families,legacy_ledger_preserved_reference=oldref,
        total_committed_PPO_updates=sum(r['PPO_updates'] for r in families),
        known_PPO_updates_lower_bound=sum(r['PPO_updates'] for r in families),
        forward_supplement='Actual successful optimizer coverage reconciled through TR10; scientific status is separate from training volume.',
        old_scientific_results_attempts_costs_valid_counts_stops_and_champion_unchanged=True,
        old_training_target_completed_declarations_are_historical_not_actual_coverage_proof=True,
        training_amount_is_not_strength_evidence=True)
    atomic_json(output/'CURRENT_VOLUME.proposed.json',proposal)
    report=dict(at=utcnow(),complete=True,engineering_only=True,optimizer_steps=0,
        unique_actual_training_decisions=total,legacy_prior_reported_decisions=old['actual_unique_training_decisions'],
        actual_prior_decisions=sum(r['trained_unique_decisions'] for r in families if r['hypothesis_id']!='TR10'),
        family_evidence=details,canonical_family_count=len(families),
        epochs_restores_and_copies_never_add_decisions=True,legacy_cursor_and_all_old_scientific_outputs_unchanged=True,
        legacy_ledger=oldref,proposal=dict(path=(output/'CURRENT_VOLUME.proposed.json').relative_to(root).as_posix(),
            sha256=sha256(output/'CURRENT_VOLUME.proposed.json')))
    atomic_json(output/'metrics.json',report)
    return report
