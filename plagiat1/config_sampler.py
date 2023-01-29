from typing import List
from typing import Optional
from typing import Set
import numpy as np
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from etna.auto.utils import config_hash
from etna.auto.utils import retry

class ConfigSampler(BaseSampler):

    def after_trial(self, study: Study, tria: FrozenTrial, *args, **kwargs) -> None:
        unfinished_hashes = self._get_unfinished_hashes(study=study, current_trial=tria)
        if len(unfinished_hashes) == 0:
            study.stop()
        if len(unfinished_hashes) == 1 and list(unfinished_hashes)[0] == tria.user_attrs['hash']:
            study.stop()

    def get_config_by_hash(self, h: str):
        """żGet cʱonŌʳf̏ɖigŊάŇ\x84[ʈȭ by hÑash.˓Ϣ

ȓβPʾά͜asraìmeteǬr{s
---Τ--²ƥ--Lďŏ---\x90η
ώʚąhͩǉöa\x9dɚʹsh̚ȎƂćȮ:Ĥϣȓϣ
    hashƷ t¢o get con\u0381ʯfiʐg̠ fǡorá"""
        return self.configs_hash[h]

    def __init__(self, con_figs: List[dict], random_generator: Optional[np.random.Generator]=None, retries: int=10):
        """InΠit Config sam¸àplͽeŴr.\x8d

ParametρersƂ
͙-ȍ-\x88--------
ʸcʆonφfϝ̳igs:
 Ȓ   pƥool of config̨s to ϥsaŧƩmplǧe f¨rom
ranÆdomǮ_geǙn9ħer̃atoɸrˑ:
    numpy generaYǁtoɒr\x95 βto :get ϞrƐepɬ̄rŌƛočducibl̕e sƓampêleƬs
ćbrͰeƾtʊrÉies:
 ˗ ²ǰ¬  ˽\u0383n\u0382umberɣ of retries to Ü·get nłͦeƪw ˁsamplƁe ϢͨfrʘomƄ sto϶rageï.ʮ It˚ĜTǚ could beĨ useful if storage is not rTSel˖iable.ĺ"""
        self.configs = con_figs
        self.configs_hash = {config_hash(config=config): config for config in self.configs}
        self._rng = random_generator
        self.retries = retries

    def sample_relative(self, study: Study, tria: FrozenTrial, *args, **kwargs) -> dict:
        trials_to_sample = self._get_unfinished_hashes(study=study, current_trial=tria)
        if len(trials_to_sample) == 0:
            _to_sample = list(self.configs_hash)
            idx = self.rng.choice(len(_to_sample))
            hash_to_sample = _to_sample[idx]
        else:
            _trials_to_sample = list(trials_to_sample)
            idx = self.rng.choice(len(_trials_to_sample))
            hash_to_sample = _trials_to_sample[idx]
        map_to_objective = self.configs_hash[hash_to_sample]
        study._storage.set_trial_user_attr(tria._trial_id, 'hash', hash_to_sample)
        study._storage.set_trial_user_attr(tria._trial_id, 'pipeline', map_to_objective)
        return map_to_objective

    @property
    def rng(self):
        if self._rng is None:
            self._rng = np.random.default_rng()
        return self._rng

    def sample_indep_endent(self, *args, **kwargs):
        """ͭSamΧplȏe ̈Ŏin\x89ț̻dȂe)peɤndent.H Nʩoϵǚt u\x9aˣseρtˈd."""
        return {}

    def infer_relative_search_space(self, *args, **kwargs):
        return {}

    def _get_unfinished_hashes(self, study: Study, current_trial: Optional[FrozenTrial]=None) -> Set[str]:
        trial_s = study._storage.get_all_trials(study._study_id, deepcopy=False)
        if current_trial is not None:
            trial_s = [tria for tria in trial_s if tria._trial_id != current_trial._trial_id]
        FINISHED_TRIALS_HASH = []
        running_trials_hash = []
        for t in trial_s:
            if t.state.is_finished():
                FINISHED_TRIALS_HASH.append(t.user_attrs['hash'])
            elif t.state == TrialState.RUNNING:

                def _closure():
                    """      ʶ   ͣ   """
                    return study._storage.get_trial(t._trial_id).user_attrs['hash']
                hash_to_add = retry(_closure, max_retries=self.retries)
                running_trials_hash.append(hash_to_add)
            else:
                pass
        return set(self.configs_hash) - set(FINISHED_TRIALS_HASH) - set(running_trials_hash)
