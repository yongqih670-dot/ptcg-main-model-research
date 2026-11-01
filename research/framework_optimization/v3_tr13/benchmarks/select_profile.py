"""Apply the pre-declared recovery-cost rule before any D measurement."""
from common import *
def main():
    if not read(V3/'reports/QUALIFICATION.json')['passed']:raise ValueError('Strict deterministic qualification required')
    target=V3/'reports/SELECTED_PROFILE.json'
    if target.exists() or list((V3/'reports').glob('D-*-final.json')):raise ValueError('No post-result selection')
    baseline=read(V3/'reports/B3-baseline-replay-cold-repair.json');candidate=read(V3/'reports/B3-candidate-replay-cold.json')
    switches=dict(control_cache=True,journal_cache=True,outbox_cache=True,capacity_transaction=candidate['seconds']<=baseline['seconds']*1.1+5,progress_coalescing=True,exit_evidence=True)
    write(target,dict(at=time.time(),switches=switches,CPU_deterministic_algorithms=True,rule_sha256=sha(V3/'PROFILE_RULE.json'),
        own_fresh_cold_cost_evidence=dict(baseline_seconds=baseline['seconds'],candidate_seconds=candidate['seconds'],maximum_candidate_seconds=baseline['seconds']*1.1+5),
        qualification_sha256=sha(V3/'reports/QUALIFICATION.json'),science_credit=0,no_server_or_scientific_adoption=True))
    print(switches)
if __name__=='__main__':main()
