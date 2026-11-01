"""Finite local forecast only, not a server ticket or old budget reset."""
from common import *
def main():
    baseline=read(V3/'reports/BASELINE.json');forecast=4*2**30
    conservative_old=max(baseline['actual_local_working_set_bytes'],int(baseline['project_budget_snapshot']['new_workspace_artifacts_gib']*2**30))
    if conservative_old+forecast>120*2**30 or shutil.disk_usage(ROOT).free-forecast<20*2**30:raise OSError('No finite local growth room')
    write(V3/'reports/STORAGE_ADMISSION.json',dict(at=time.time(),local_only=True,initial_growth_forecast_bytes=1536*2**20,new_v3_total_growth_upper_bytes=forecast,
        conservative_old_working_set_bytes=conservative_old,conservative_sum_bytes=conservative_old+forecast,hard_bytes=120*2**30,physical_free_bytes=shutil.disk_usage(ROOT).free,
        native_full_project_capacity_gate_unchanged=True,old_files_never_deleted=True,temporary_case_release_requires_full_lossless_archive_ACK_and_no_live_recovery_reference=True,
        failed_zero_update_exception='Only diagnosed B/B2 preparation cases, no recovery index or optimizer calls; full raw file manifest and verified lossless package retained. Both failure fees retained.',server_authority=False))
    print('Conservative local growth forecast within unchanged120GiB; no server scope activated')
if __name__=='__main__':main()
