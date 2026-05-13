"""skein runner — re-export of the legacy benches.runners.skein_runner.

Phase B leaves the implementation in benches/runners/ to avoid
duplicating the per-(family, penalty) dispatch table. The v2 namespace
just imports it so cell_driver can resolve packages uniformly under
`benches.v2.runners.*`.
"""
from benches.runners.skein_runner import (  # noqa: F401
    name, is_available, fit,
)
