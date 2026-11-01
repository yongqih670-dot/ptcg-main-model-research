"""Same owner-thread instrumentation on both frozen code paths; no CUDA sync."""
import ast,contextlib,functools,inspect,threading,time
class Trace:
    def __init__(self):self.started=self.last=time.monotonic();self.owner=threading.get_ident();self.stack=[];self.exclusive={};self.inclusive={};self.calls={};self.base='driver_bookkeeping';self.background_calls=0;self.bookkeeping_ns=0
    def tick(self):
        now=time.monotonic();key=self.stack[-1][0] if self.stack else self.base;self.exclusive[key]=self.exclusive.get(key,0)+now-self.last;self.last=now;return now
    @contextlib.contextmanager
    def span(self,key):
        if threading.get_ident()!=self.owner:self.background_calls+=1;yield;return
        overhead_began=time.perf_counter_ns()
        began=self.tick();self.stack.append((key,began));self.calls[key]=self.calls.get(key,0)+1
        self.bookkeeping_ns+=time.perf_counter_ns()-overhead_began
        try:yield
        finally:
            overhead_began=time.perf_counter_ns()
            now=self.tick();name,began=self.stack.pop();self.inclusive[name]=self.inclusive.get(name,0)+now-began
            self.bookkeeping_ns+=time.perf_counter_ns()-overhead_began
    def wrap(self,owner,name,key):
        original=getattr(owner,name)
        @functools.wraps(original)
        def measured(*a,**kw):
            with self.span(key):return original(*a,**kw)
        setattr(owner,name,measured)
    def paths(self,iterator):
        with self.span('object_directory_enumeration'):return list(iterator)
    def enumerate_function(self,owner,name):
        original=getattr(owner,name);module=inspect.getmodule(original);tree=ast.parse(inspect.getsource(original))
        class Instrument(ast.NodeTransformer):
            def visit_Call(self,node):
                self.generic_visit(node)
                if isinstance(node.func,ast.Attribute) and node.func.attr=='glob':return ast.copy_location(ast.Call(func=ast.Attribute(value=ast.Name(id='__v3_trace',ctx=ast.Load()),attr='paths',ctx=ast.Load()),args=[node],keywords=[]),node)
                return node
        tree=Instrument().visit(tree);ast.fix_missing_locations(tree);namespace=dict(module.__dict__,__v3_trace=self);exec(compile(tree,module.__file__+'[same-instrumentation]','exec'),namespace);setattr(owner,name,namespace[original.__name__])
    def report(self):
        end=self.tick();return dict(wall_seconds=end-self.started,exclusive_seconds=self.exclusive,inclusive_seconds_nonadditive=self.inclusive,calls=self.calls,
            partition_residual_seconds=end-self.started-sum(self.exclusive.values()),background_calls_not_attributed=self.background_calls,CUDA_synchronization=False,
            measured_instrumentation_bookkeeping_seconds=self.bookkeeping_ns/1e9,instrumentation_cost_scope='span entry/exit only; complete on/off control probe reported separately',monotonic_resolution_seconds=time.get_clock_info('monotonic').resolution,
            original_production_spans_overlapping_do_not_sum=self.production_timer.report() if hasattr(self,'production_timer') else None)

def install(t):
    import ptcg_lab.framework_v2_training as native
    import ptcg_lab.training_control as control
    import ptcg_lab.archive_outbox as outbox
    import ptcg_lab.bounded_io as writer
    import ptcg_lab.fo_v2.durable as durable
    import ptcg_lab.fo_v2.checkpoint as checkpoint
    import ptcg_lab.server_workspace as workspace
    import ptcg_lab.io as io
    import torch
    import ptcg_lab.phase_timing as production_timing
    # Reuse the exact production span instrumentation for pacing, writes,
    # fsync and ownership checks. Its report overlaps this report: do not sum.
    if production_timing._timer is not None:raise ValueError('Existing production timer must not be overwritten')
    production_timing._timer=production_timing.Timer();t.production_timer=production_timing._timer
    # Materialize only glob iterators that the original function immediately
    # sorts; no action, RNG, iteration order or arithmetic changes.
    t.enumerate_function(native,'advance_native_state')
    for owner,name,key in [(native.NativeCheckpoints,'save','checkpoint_transaction'),(native,'advance_native_state','journal_advance'),
        (native,'semantic_digest','complete_state_semantic_hash'),(native.NativeCheckpoints,'checked','canonical_identity_and_hash'),
        (native.DriverCheckpoints,'stage','immutable_stage'),(native.ProjectCapacity,'_scan','whole_project_capacity_scan'),
        (durable.Capacity,'reconcile','capacity_reconciliation'),(durable.Capacity,'reserve','capacity_reservation'),(durable.Capacity,'_persist','capacity_ledger_publication'),
        (checkpoint.Checkpoints,'_validate_index','recovery_graph_validation'),(checkpoint.Checkpoints,'commit','tail_and_descriptor_commit'),
        (control.TrainingControl,'wait','control_poll'),(control.TrainingControl,'request','control_query'),(control,'read_json','control_file_read_decode'),
        (native,'read_json','journal_file_read_decode'),(outbox.ArchiveOutbox,'records','queue_scan'),(outbox.ArchiveOutbox,'_ack','ACK_lookup'),
        (outbox.ArchiveOutbox,'queue_recovery','queue_checkpoint_transaction'),(outbox.ArchiveOutbox,'pending','ACK_pending_query'),
        (outbox,'read_json','queue_file_read_decode'),(writer.WriteContext,'atomic_write','atomic_metadata_or_legacy_publication'),
        (workspace,'admit','whole_workspace_admission'),(io,'atomic_json','json_publication'),(torch,'save','torch_serialization'),
        (durable.HashEvidence,'verify','immutable_reference_verification'),(durable.Objects,'put','immutable_object_transaction')]:t.wrap(owner,name,key)
    native.advance_native_state.__globals__['read_json']=native.read_json
