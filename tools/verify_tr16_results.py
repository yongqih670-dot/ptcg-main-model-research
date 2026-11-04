"""Offline verification of TR16 compact BO3 outcomes; no training or model loading."""
import json
from verify_tr14_tr15_results import verify,read
if __name__ == '__main__':
    result=verify('TR16')
    endpoint=read('research/autonomy/TR16_REGISTERED_ENDPOINT.json')
    assert endpoint['version']==107 and endpoint['actual_unique']==839120
    assert endpoint['new_unique']==431968 and endpoint['pending_unoptimized']==4174
    assert endpoint['training_stopped'] and not endpoint['original_target_complete']
    print(json.dumps(result,ensure_ascii=False))
