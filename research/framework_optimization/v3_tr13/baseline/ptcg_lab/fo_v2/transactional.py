"""Production-facing transaction reserve wrapper around immutable Objects.

The local pilot uses the base store with ample admitted headroom. This wrapper
also proves the near-capacity case: .partial and final hardlink coexist briefly.
It adds ledger transactions, not model/PPO changes. Its incremental cost must be
included in the server difference acceptance before native production use.
"""
from .durable import Objects


class TransactionalObjects(Objects):
    def put(self,serialize,maximum,kind='blob'):
        # Base put reserves the temporary write. This reserves the additional
        # logical pathname during exclusive hardlink publication as well.
        publication=self.capacity.reserve(maximum)
        try:
            return super().put(serialize,maximum,kind)
        except BaseException:
            # E.g. link succeeds but unlink is denied: both owned names remain.
            # Rebuild actual accounting; never assume failed bytes disappeared.
            self.capacity.reconcile()
            raise
        finally:self.capacity.finish(publication,0)
