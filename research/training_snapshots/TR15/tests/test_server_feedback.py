"""Synthetic protocol cases, never experimental or leaderboard results."""
import copy
import unittest
from ptcg_lab.server_feedback import assess_feedback,replay_checks,PUBLIC,VALIDATION


def episode(eid,kind=PUBLIC,other=222):
    meta={'id':eid,'state':'COMPLETED','type':kind,'end_time':'2026-10-23T10:00:00Z',
          'agents':[{'index':0,'submission_id':111,'team_id':11,'state':'EPISODE_AGENT_STATE_UNSPECIFIED','reward':1},
                    {'index':1,'submission_id':other,'team_id':11 if other==111 else 22,'state':'EPISODE_AGENT_STATE_UNSPECIFIED','reward':-1}]}
    replay={'name':'cabt','info':{'EpisodeId':eid},'module_version':'1.32.7',
            'configuration':{'version':2,'bo':3},'statuses':['DONE','DONE'],'rewards':[1,-1],
            'steps':[[{'status':'ACTIVE','visualize':[{'current':{'result':0}},{'current':{'result':0}}]}, {'status':'ACTIVE'}],
                     [{'status':'DONE','reward':1},{'status':'DONE','reward':-1}]]}
    return meta,replay


class FeedbackTests(unittest.TestCase):
    def setup_evidence(self,n=20):
        pairs=[episode(i) for i in range(1,n+1)]+[episode(100,VALIDATION,111)]
        score={'submission_id':111,'score':'1000.0','source':'authenticated submission API snapshot','captured_at':'2026-10-23T11:00:00Z'}
        return [x[0] for x in pairs],{x[0]['id']:x[1] for x in pairs},score
    def test_valid_server_evidence_still_not_full_goal(self):
        metadata,replays,score=self.setup_evidence()
        result=assess_feedback(111,metadata,replays,score)
        self.assertTrue(result['server_feedback_verified']);self.assertFalse(result['goal_complete'])
        self.assertEqual(result['normal_verified_count'],20)
    def test_self_games_exhibitions_and_duplicates_do_not_fill_minimum(self):
        metadata,replays,score=self.setup_evidence(19)
        extra,replay=episode(30,PUBLIC,111);metadata+=[extra,metadata[0]];replays[30]=replay
        extra,replay=episode(31,'EPISODE_TYPE_EXHIBITION');metadata.append(extra);replays[31]=replay
        result=assess_feedback(111,metadata,replays,score)
        self.assertEqual(result['normal_verified_count'],19);self.assertFalse(result['server_feedback_verified'])
    def test_missing_replay_or_wrong_bo_or_identity_cannot_count(self):
        for change in ('missing','bo','id','unfinished','missing_end_time'):
            metadata,replays,score=self.setup_evidence()
            if change=='missing':replays.pop(1)
            elif change=='bo':replays[1]['configuration']['bo']=1
            elif change=='id':replays[1]['info']['EpisodeId']=999
            elif change=='unfinished':replays[1]['steps'][0][0]['visualize']=[{'current':{'result':0}},{'current':{'result':1}}]
            else:metadata[0]['end_time']=None
            self.assertFalse(assess_feedback(111,metadata,replays,score)['server_feedback_verified'])
    def test_failed_episode_retained_even_with_twenty_good_games(self):
        metadata,replays,score=self.setup_evidence()
        bad,replay=episode(50);bad['state']='ERRORED';replay['steps'][-1][0]['status']='TIMEOUT'
        metadata.append(bad);replays[50]=replay
        result=assess_feedback(111,metadata,replays,score)
        self.assertEqual(result['normal_verified_count'],20);self.assertEqual(result['explicit_failed_episode_ids'],[50])
        self.assertFalse(result['server_feedback_verified'])
    def test_stale_or_wrong_score_and_conflicting_duplicates_fail(self):
        for change in ('stale','wrong_submission','conflict','nan'):
            metadata,replays,score=self.setup_evidence()
            if change=='stale':score['captured_at']='2026-10-22T11:00:00Z'
            elif change=='wrong_submission':score['submission_id']=222
            elif change=='nan':score['score']='NaN'
            else:
                changed=copy.deepcopy(metadata[0]);changed['agents'][0]['reward']=-1;metadata.append(changed)
            self.assertFalse(assess_feedback(111,metadata,replays,score)['server_feedback_verified'])
    def test_threshold_cannot_be_reduced(self):
        with self.assertRaises(ValueError):assess_feedback(111,[],{},None,minimum_normal=19)


if __name__=='__main__':unittest.main()
