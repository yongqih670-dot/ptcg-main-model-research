"""Materialize ONLY the selected public frozen replay, with original-byte proof."""
import gzip,hashlib,json,os,sys,time
from pathlib import Path
V3=Path(__file__).resolve().parents[1];ROOT=V3.parents[2];sys.path.insert(0,str(V3/'baseline'))
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def main():
    began=time.monotonic();target=V3/'shared-inputs';target.mkdir(exist_ok=True);report=V3/'reports/INPUTS.json'
    if report.exists():raise ValueError('Inputs already frozen; verify, do not overwrite')
    old=json.loads((ROOT/'research/framework_optimization/v2/reports/FIXTURE.json').read_text(encoding='utf-8'))
    refine=json.loads((ROOT/'research/framework_optimization/v2/reports/FIXTURE_REFINEMENT.json').read_text(encoding='utf-8'))
    training=json.loads((ROOT/'runs/20261030T142634Z_tr13-volume-segment-004_53eca8/remote-results/payload-config.json').read_text(encoding='utf-8'))
    if old['ppo']!=training['ppo'] or old['BO3']!=48 or old['rollout_decisions']!=8203:raise ValueError('Frozen replay not applicable to current PPO')
    from ptcg_lab.archive_compressed import verify_archive
    refs=[];catalog={}
    def add(p,sha,size):
        if p.stat().st_size!=size or digest(p)!=sha:raise ValueError('Selected original bytes differ: '+str(p))
        row={'path':str(p.relative_to(ROOT)).replace('\\','/'),'sha256':sha,'bytes':size};catalog[sha]=row;return row
    for item in old['trajectories']:
        derived={}
        for name,spec in [('trajectory',item),('summary',item['summary_archive'])]:
            canonical=item['summary']['trajectory_path'] if name=='trajectory' else f"series/{item['series_id']:06d}.json"
            archive=ROOT/spec['archive_root'];ack=spec.get('ack_value')
            if ack:verify_archive(archive,ack)
            dst=target/canonical;dst.parent.mkdir(parents=True,exist_ok=True)
            if not dst.exists():
                source=archive/spec['archive_relative']
                with gzip.open(source,'rb') as src,dst.open('xb') as out:
                    total=0;h=hashlib.sha256()
                    while b:=src.read(2**20):
                        total+=len(b)
                        if total>spec['raw_bytes']:raise ValueError('Expanded replay exceeds exact original byte count')
                        out.write(b);h.update(b)
                    out.flush();os.fsync(out.fileno())
                if total!=spec['raw_bytes'] or h.hexdigest()!=spec['object_sha256']:raise ValueError('Replay original hash differs')
            derived[name]=add(dst,spec['object_sha256'],spec['raw_bytes'])
        refs.append({'series_id':item['series_id'],'control_calls':item['summary']['engine_decisions'],'summary_value':{k:v for k,v in item['summary'].items() if k!='sample_observations'},**derived})
    source=ROOT/'runs/20261028T110225Z_tr12-full-tail-GPU-acceptance_f85b47';archive=source/'online-archive'
    index=json.loads((source/'remote-results/checkpoint-commits/index.json').read_text(encoding='utf-8'));cp=index['commits'][-1]['checkpoint']
    ack=next(json.loads(p.read_text(encoding='utf-8')) for p in (archive/source.name/'acks').glob('*.json') if json.loads(p.read_text(encoding='utf-8'))['sha256']==cp['sha256'])
    raw=verify_archive(archive,ack,collect=True);parent=target/'parent.pt'
    if not parent.exists():
        with parent.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    parent_ref=add(parent,cp['sha256'],cp['bytes'])
    import torch
    torch.set_num_threads(2);value=torch.load(parent,map_location='cpu',weights_only=False)
    league_path=ROOT/refine['league_path'];league=json.loads(league_path.read_text(encoding='utf-8'));add(league_path,refine['league_sha256'],league_path.stat().st_size)
    origin=f"checkpoint-commits/objects/prefix-{value['archive_origin_prefix_sha256']}.json";origin_path=league_path.parent/origin
    canonical_refs=[{'canonical':'prefix.json',**add(origin_path,value['archive_origin_prefix_sha256'],origin_path.stat().st_size)},
                    {'canonical':'league.json',**catalog[refine['league_sha256']]}]
    for row in league['snapshots']:canonical_refs.append({'canonical':row['path'],**add(league_path.parent/row['path'],row['sha256'],(league_path.parent/row['path']).stat().st_size)})
    state=json.loads((ROOT/'research/autonomy/state.json').read_text(encoding='utf-8'))
    if state.get('active_job'):raise RuntimeError('Engineering yields to actual active job')
    result=dict(parent=parent_ref,canonical_refs=canonical_refs,trajectories=refs,immutable_catalog=catalog,ppo=old['ppo'],
        complete_BO3=48,replay_decisions=8203,control_calls=sum(r['control_calls'] for r in refs),
        input_ready_archive_set='same frozen canonical trajectory/summary originals; each case mints own run/epoch ACKs before timing, never uses previous case outputs',
        CUDA_RNG='all original payloads preserved without CUDA execution',current_bundle=training['bundle_path'],
        actor_source_sha256=old['actor_source_sha256'],logical_update_timestamp='engineering-frozen-input-clock-not-a-scientific-result',
        source_fixture_sha256=digest(ROOT/'research/framework_optimization/v2/reports/FIXTURE.json'),
        materialization_wall_seconds=time.monotonic()-began,scientific_decisions=0)
    report.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:result[k] for k in ('complete_BO3','replay_decisions','control_calls','materialization_wall_seconds')}))
if __name__=='__main__':main()
