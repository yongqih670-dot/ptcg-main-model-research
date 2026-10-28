"""Original bytes for exact forensic replicas and explicitly archived closed rollouts."""
import gzip,hashlib,os,re
from pathlib import Path
from .io import read_json,sha256
from .archive_ack import publication_path,validate_ack,safe_target,archive_root
from .local_cache_retirement import windows_security,identity,digest_stream

ARCHIVES={'20261025T162703Z_tr04-failure-archive_c35193',
          '20261025T192352Z_tr05-failure-archive_c69021','20261025T211209Z_tr06-archive_5deb4f'}
TR07_ARCHIVE='20261026T015921Z_tr07-archive-bounded-connect_4cb622'
TR08_FIRST_ARCHIVE='20261026T104756Z_tr08-first-terminal-full-archive_c60e40'
TARGET_ARCHIVES=ARCHIVES|{TR07_ARCHIVE,TR08_FIRST_ARCHIVE}
TR03_SOURCE='20261023T151043Z_tr03-ppo-262k_a347a7'


def owned_path(root,path,tokens,owner):
    root=Path(publication_path(root)).absolute();path=Path(publication_path(path)).absolute()
    if root not in path.parents or path!=path.resolve():raise ValueError('Evidence real path escaped the project')
    allowed=set(tokens.values())
    if set(tokens)!={'desktop','current_Codex_sandbox'}:raise ValueError('Unrecognized local execution identities')
    parts=path.relative_to(root).parts
    for candidate in [root,*[root.joinpath(*parts[:n]) for n in range(1,len(parts)+1)]]:
        st=candidate.lstat()
        if candidate.is_symlink() or getattr(st,'st_file_attributes',0)&0x400:
            raise ValueError('Evidence path traverses a reparse point')
        observed=hashlib.sha256(owner(candidate).encode()).hexdigest()
        if observed not in allowed or candidate==root and observed!=tokens['desktop']:
            raise ValueError('Evidence owner is neither the verified project user nor current Codex execution identity')
    return path


def target_relative(relative):
    parts=Path(relative).parts
    if len(parts)==4 and parts[:3]==('runs',TR03_SOURCE,'rollouts') and re.fullmatch(r'v[0-9]{4}-s[0-9]{6}\.pt',parts[3]):
        return Path(*parts).as_posix()
    if len(parts)<4 or parts[:1]!=('runs',) or parts[1] not in TARGET_ARCHIVES or parts[2]!='archive' or not str(relative).endswith('.pt') or '..' in parts:
        raise ValueError('Only named closed-hypothesis forensic tensor replicas may be tiered')
    return Path(*parts).as_posix()


def source_blocks(root,row,tokens,owner,pacer=None):
    root=Path(root);ref=row['retained_source'];digest=hashlib.sha256();count=0
    if not 0<row['bytes']<=128*2**20:raise ValueError('Original evidence object exceeds its bound')
    if ref['kind']=='retained_primary_raw':
        relative=Path(ref['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Primary evidence path escaped')
        if len(relative.parts)>=3 and relative.parts[0]=='runs' and relative.parts[1] in TARGET_ARCHIVES:
            raise ValueError('Retained primary evidence cannot depend on another retirement target')
        path=owned_path(root,root/relative,tokens,owner)
        if identity(path)!=ref['file_identity'] or path.stat().st_size!=row['bytes'] or ref['sha256']!=row['sha256']:
            raise ValueError('Retained primary evidence identity differs')
        stream=path.open('rb')
    elif ref['kind']=='local_exact_gzip':
        relative=Path(ref['path'])
        if relative.is_absolute() or '..' in relative.parts or len(relative.parts)!=4 or relative.parts[0]!='runs' or relative.parts[2]!='lossless-gzip' or relative.parts[3]!=row['sha256']+'.pt.gz':
            raise ValueError('Local exact gzip namespace differs')
        path=owned_path(root,root/relative,tokens,owner);before=identity(path)
        with path.open('rb') as raw:
            if pacer:encoded,size=digest_stream(raw,pacer)
            else:encoded,size=sha256(path),path.stat().st_size
        if encoded!=ref['sha256'] or size!=ref['bytes'] or identity(path)!=before:
            raise ValueError('Local encoded original changed')
        stream=gzip.open(path,'rb')
    elif ref['kind']=='durable_gzip_ACK':
        ar=(root/ref['archive_root']).resolve()
        if root.resolve() not in ar.parents:raise ValueError('Durable source archive escaped the project')
        ack=ref['ACK'];validate_ack(ack,ack['run_id'],ack['lease_epoch'],ack['object'],row['sha256'],row['bytes'],ack['sequence'])
        if ack['storage']['codec']!='gzip':raise ValueError('Only verified original-byte gzip sources are permitted')
        expected=f"{ack['run_id']}/objects/{ack['object']}.gz"
        if ack['archive_relative']!=expected:raise ValueError('Durable original archive namespace differs')
        path=owned_path(root,safe_target(archive_root(ar),expected),tokens,owner)
        before=identity(path)
        with path.open('rb') as raw:
            if pacer:encoded,size=digest_stream(raw,pacer)
            else:
                encoded=sha256(path);size=path.stat().st_size
        if encoded!=ack['storage']['sha256'] or size!=ack['storage']['bytes'] or identity(path)!=before:
            raise ValueError('Durable compressed original changed')
        stream=gzip.open(path,'rb')
    else:raise ValueError('Unknown original evidence storage kind')
    before=identity(path)
    with stream:
        while block:=stream.read(1024*1024):
            if pacer:pacer.consume(len(block))
            count+=len(block)
            if count>row['bytes']:raise ValueError('Restored original exceeds declared evidence length')
            digest.update(block);yield block
    if count!=row['bytes'] or digest.hexdigest()!=row['sha256'] or identity(path)!=before:
        raise ValueError('Original evidence full hash or file identity differs')


def original_bytes(root,row,tokens,owner,pacer=None):
    return b''.join(source_blocks(root,row,tokens,owner,pacer))


def read_missing(path):
    """Only exact named archival paths use the append-only location catalog."""
    if os.name!='nt':return None
    path=Path(publication_path(path)).absolute()
    for root in path.parents:
        catalog=root/'research/autonomy/evidence-locations.json'
        if not catalog.is_file():continue
        relative=path.relative_to(root).as_posix()
        try:target_relative(relative)
        except ValueError:return None
        data=read_json(catalog)
        if str(root)!=publication_path(data['project_path']):raise ValueError('Evidence location catalog project differs')
        entry=data['objects'].get(relative)
        if entry is None:return None
        manifest=owned_path(root,root/entry['manifest']['path'],data['trusted_execution_tokens'],windows_security()[1])
        if sha256(manifest)!=entry['manifest']['sha256']:raise ValueError('Durable evidence tiering manifest changed')
        proof=read_json(manifest);rows=[r for r in proof['rows'] if r['target']==relative]
        if len(rows)!=1:raise ValueError('Original evidence locator is missing or duplicated')
        _,owner=windows_security()
        return original_bytes(root,rows[0],data['trusted_execution_tokens'],owner)
    return None
