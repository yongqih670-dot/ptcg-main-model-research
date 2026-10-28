"""Transactional lossless archival of a bounded closed TR03 rollout inventory."""
import gzip,hashlib,os,re,time
from pathlib import Path

from .io import atomic_json,read_json,sha256,utcnow
from .evidence_tiering import TR03_SOURCE,target_relative,owned_path,source_blocks,read_missing
from .local_cache_retirement import windows_security,identity,LocalReadPacer,digest_stream
from .local_forensic_retirement import delete_access


def evidence_digest(path,pacer=None):
    if pacer is None:return sha256(path)
    with path.open('rb') as stream:return digest_stream(stream,pacer)[0]


def check_scope(root,config,inventory,pacer=None):
    if config.get('source_run_id')!=TR03_SOURCE or not 1<=len(inventory['rows'])<=256:
        raise ValueError('Only at most256 exact closed TR03 rollout originals may archive')
    state=read_json(root/'research/autonomy/state.json');h=state['research_stage']['hypotheses']['TR03']
    source=root/'runs'/TR03_SOURCE
    if h.get('status')!='completed' or h.get('classification')!='inconclusive' or TR03_SOURCE not in h['run_ids']:
        raise ValueError('TR03 must remain completed and closed')
    metrics=read_json(source/'metrics.json')
    if not metrics.get('complete') or metrics.get('updates')!=32 or metrics.get('trained_decisions')!=262165:
        raise ValueError('Complete original TR03 learning output differs')
    if sum(r['bytes'] for r in inventory['rows'])>3*2**30 or len({r['target'] for r in inventory['rows']})!=len(inventory['rows']):
        raise ValueError('Bounded unique original trajectory inventory differs')
    for relative,digest in inventory['protected_evidence'].items():
        parts=Path(relative).parts
        if len(parts)<3 or parts[:2]!=('runs',TR03_SOURCE) or '..' in parts:
            raise ValueError('Protected evidence path escaped the closed TR03 source')
        if evidence_digest(root/relative,pacer)!=digest:raise ValueError('Protected original model/checkpoint/metadata changed')
    for row in inventory['rows']:
        relative=target_relative(row['target']);parts=Path(relative).parts
        if parts[:3]!=('runs',TR03_SOURCE,'rollouts') or type(row['bytes']) is not int or not 0<row['bytes']<=128*2**20 or not re.fullmatch('[0-9a-f]{64}',row['sha256']):
            raise ValueError('Only exact original TR03 rollout shards may archive')
        sid=int(Path(relative).stem.split('-s')[1]);summary=source/'series'/f'{sid:06d}.json'
        data=read_json(summary)
        if sha256(summary)!=row['summary_sha256'] or data['statuses']!=['DONE','DONE'] or data['series_id']!=sid or not 0<=int(data['policy_version'])<32 or int(data['policy_version'])!=int(parts[3][1:5]) or data['trajectory_sha256']!=row['sha256'] or data['trajectory_path'].replace('\\','/')!=str(Path(*parts[2:]).as_posix()):
            raise ValueError('Only consumed complete BO3 original bytes may archive')


def pack_original(path,packed,expected,raw_bytes,pacer):
    """Publish an encoded file only after the entire input matches its old SHA."""
    if packed.exists():raise ValueError('Encoded destination already exists')
    temporary=packed.with_suffix(packed.suffix+'.partial');digest=hashlib.sha256();count=0
    with path.open('rb') as source,temporary.open('xb') as raw:
        with gzip.GzipFile(filename='',fileobj=raw,mode='wb',compresslevel=1,mtime=0) as target:
            while block:=source.read(1024*1024):
                pacer.consume(len(block));count+=len(block);digest.update(block);target.write(block)
                if count>raw_bytes:raise ValueError('Original input grew during compression')
        raw.flush();os.fsync(raw.fileno())
    if count!=raw_bytes or digest.hexdigest()!=expected:raise ValueError('Original full hash differs; never unlink it')
    os.link(temporary,packed);temporary.unlink()
    return {'kind':'local_exact_gzip','path':None,'sha256':sha256(packed),'bytes':packed.stat().st_size}


def run(config,output):
    root=Path(config['project_path']).absolute();output=Path(output).resolve()
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name or output.parent!=root/'runs':raise RuntimeError('Controlled local runner required')
    if (config.get('job_kind'),config.get('retirement_variant'),config.get('maximum_attempts'))!=('local_cache_retirement','consumed_trajectory_gzip',1):
        raise ValueError('Unexpected exact lossless original archival role')
    if sha256(Path(__file__))!=config['retirement_module_sha256']:raise ValueError('Pinned original archival source changed')
    deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX'])
    if not 0<deadline-time.time()<=900.5:raise ValueError('One bounded900s archival scope only')
    spec=config['inspection_evidence'];proof=root/spec['path']
    if sha256(proof)!=spec['sha256']:raise ValueError('Exact original inventory changed')
    inventory=read_json(proof);pacer=LocalReadPacer(16*2**20,deadline-25);check_scope(root,config,inventory,pacer);tokens=inventory['trusted_execution_tokens'];sid,owner=windows_security()
    if hashlib.sha256(sid.encode()).hexdigest()!=tokens['desktop']:raise ValueError('Expected actual project-user identity')
    packed_root=output/'lossless-gzip';packed_root.mkdir()
    catalog_path=root/'research/autonomy/evidence-locations.json';catalog=read_json(catalog_path)
    if catalog['project_path']!=str(root) or catalog['trusted_execution_tokens']!=tokens:raise ValueError('Existing exact-byte catalog identity differs')
    total=0;encoded=0
    for number,row in enumerate(inventory['rows']):
        if time.time()>deadline-30:raise TimeoutError('Preserve completed individual transactions and untouched raw remainder')
        path=owned_path(root,root/row['target'],tokens,owner)
        if identity(path)!=row['file_identity'] or hashlib.sha256(owner(path).encode()).hexdigest()!=row['owner_token_sha256'] or row['target'] in catalog['objects']:
            raise ValueError('Unchanged original OS file identity required')
        delete_access(path);packed=packed_root/(row['sha256']+'.pt.gz')
        ref=pack_original(path,packed,row['sha256'],row['bytes'],pacer);ref['path']=packed.relative_to(root).as_posix()
        archived={**row,'retained_source':ref}
        for _ in source_blocks(root,archived,tokens,owner,pacer):pass
        if identity(path)!=row['file_identity']:raise ValueError('Original changed during verified compression')
        entry=output/'entries'/f'{number:04d}.json'
        atomic_json(entry,{'at':utcnow(),'rows':[archived],'trusted_execution_tokens':tokens,'all_original_bytes_and_encoded_bytes_verified':True})
        manifest={'path':entry.relative_to(root).as_posix(),'sha256':sha256(entry)}
        catalog['objects'][row['target']]={'sha256':row['sha256'],'bytes':row['bytes'],'manifest':manifest}
        atomic_json(catalog_path,catalog)
        # The original remains intact until the durable locator and full raw
        # reconstruction exist. The last fresh hash rejects a concurrent edit.
        with path.open('rb') as original:digest,size=digest_stream(original,pacer)
        if (digest,size)!=(row['sha256'],row['bytes']) or identity(path)!=row['file_identity']:
            raise ValueError('Original changed immediately before exact owned-file unlink')
        atomic_json(output/'receipts'/f'{number:04d}.json',{'status':'intent','target':row['target'],'manifest':manifest})
        path.unlink();total+=row['bytes'];encoded+=ref['bytes']
        atomic_json(output/'receipts'/f'{number:04d}.json',{'status':'archived','target':row['target'],'manifest':manifest})
        atomic_json(output/'progress.json',{'at':utcnow(),'stage':'lossless_original_archival','files':number+1,'original_bytes':total,'encoded_bytes':encoded})
    # Exercise the actual production read-through on every archived original.
    for row in inventory['rows']:
        data=read_missing(root/row['target'])
        if data is None or len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
            raise ValueError('Actual post-archival original byte read-through failed')
        pacer.consume(len(data))
    for relative,digest in inventory['protected_evidence'].items():
        if evidence_digest(root/relative,pacer)!=digest:raise ValueError('Original evidence changed')
    atomic_json(output/'metrics.json',{'at':utcnow(),'complete':True,'engineering_only':True,'source_run_id':TR03_SOURCE,
        'archived_original_files':len(inventory['rows']),'original_bytes':total,'encoded_bytes':encoded,
        'logical_bytes_released':total-encoded,'all_original_encoded_and_reconstructed_full_SHA_verified':True,
        'actual_production_read_through_all_files_passed':True,'all_model_checkpoint_league_summary_and_old_cost_evidence_unchanged':True,
        'no_training_evaluation_or_strength_selection':True,'promotion_eligible':False})
